#!/usr/bin/env -S uv run --script --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "yt-dlp>=2025.1.1",
#   "mutagen>=1.47",
#   "demucs>=4.1.0",
#   "numpy",                     # demucs 4.1.0 importa numpy mas não declara
#   "melband-roformer-infer>=0.1.5",
#   "ml-collections",
#   "soundfile",
#   "pyyaml",
#   "huggingface-hub",
# ]
# ///
"""
yt-audio — baixa vídeos do YouTube como áudio e separa os instrumentos.

Uso:
    ./yt_audio.py <url> [<url> ...]          # baixa direto
    ./yt_audio.py                            # modo interativo (cole links)
    ./yt_audio.py <url> --format best        # mantém o codec original (opus)
    ./yt_audio.py <url> -o ~/Downloads       # muda a pasta de saída

    ./yt_audio.py <url> --stems              # + separa os instrumentos
    ./yt_audio.py <url> --stems --stems-mode guitar   # guitarra + base
    ./yt_audio.py <url> --stems --no-guitar  # só a base, sem guitarra
    ./yt_audio.py --stems-only <arquivo.mp3>  # separa um arquivo local

Por padrão baixa o melhor stream de áudio disponível e converte para
MP3 320 kbps (a qualidade máxima do formato). Use --format best para
manter o codec original do YouTube sem re-encode.

Com --stems, roda separação de instrumentos local e gratuita (ver stems.py):
htdemucs_6s dá drums/bass/vocals/guitar/piano/other, e um MelBand-Roformer
dedicado refaz a guitarra com qualidade bem melhor. Sai também no_guitar.wav,
a base sem guitarra para tocar em cima.
"""

from __future__ import annotations

import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yt_dlp

SCRIPT_DIR = Path(__file__).resolve().parent
# console.py e stems.py ficam ao lado deste arquivo; garante o import mesmo rodando de outra pasta
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from console import ColorFormatter, progress  # noqa: E402

# fora de um cover o download é avulso; músicas de cover vivem em COVERS_DIR
DEFAULT_OUTPUT_DIR = Path.home() / "Downloads"
COVERS_DIR = Path.home() / "Music" / "Covers"
COVER_SUBDIRS = ("song and stems", "garageband project", "exported mixes", "final video")
AUDIO_SUBDIR = COVER_SUBDIRS[0]
LOG_FILE = SCRIPT_DIR / "yt_audio.log"
STEMS_MODES = ("full", "6s", "guitar", "noguitar")

# ---------------------------------------------------------------- logging ---


def setup_logging(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("yt-audio")
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(ColorFormatter())
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    )
    logger.addHandler(file_handler)

    # avisos do torch/demucs poluem o terminal; vão só para o arquivo
    logging.captureWarnings(True)
    py_warnings = logging.getLogger("py.warnings")
    py_warnings.addHandler(file_handler)
    py_warnings.propagate = False
    return logger


class YdlLogAdapter:
    """Redireciona o log interno do yt-dlp para o nosso logger."""

    def __init__(self, logger: logging.Logger):
        self._log = logger

    def debug(self, msg: str) -> None:
        # yt-dlp manda tanto debug quanto info para .debug(); separa pelo prefixo
        if msg.startswith("[debug] "):
            self._log.debug(msg.removeprefix("[debug] "))
        else:
            self._log.debug(msg)

    def info(self, msg: str) -> None:
        self._log.debug(msg)

    def warning(self, msg: str) -> None:
        self._log.warning(msg)

    def error(self, msg: str) -> None:
        self._log.error(msg)


# --------------------------------------------------------------- download ---

def make_progress_hook(logger: logging.Logger):
    def hook(d: dict) -> None:
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            speed = d.get("speed")
            eta = d.get("eta")
            detail = " · ".join(filter(None, [
                f"{done / 1e6:.1f}/{total / 1e6:.1f} MB" if total else None,
                f"{speed / 1e6:.1f} MB/s" if speed else None,
                f"ETA {eta}s" if eta is not None else None,
            ]))
            progress("download", done / total if total else 0.0, detail)
        elif d["status"] == "finished":
            progress("download", 1.0, final=True)
            logger.info("Download concluído, extraindo áudio…")

    return hook


def build_ydl_opts(
    output_dir: Path, audio_format: str, logger: logging.Logger, filename: str | None = None
) -> dict:
    postprocessors = [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": audio_format,  # "best" = sem re-encode
            # mp3: 320 kbps CBR (máximo do formato); demais: melhor qualidade
            "preferredquality": "320" if audio_format == "mp3" else "0",
        },
        {"key": "FFmpegMetadata", "add_metadata": True},
    ]
    return {
        "format": "bestaudio/best",
        # O player web padrão exige resolver um desafio JS que nem sempre está
        # disponível aqui; sem isso os formatos servidos respondem 403.
        # O player visionos entrega o áudio sem passar por esse desafio.
        "extractor_args": {"youtube": {"player_client": ["visionos"]}},
        "outtmpl": str(output_dir / f"{filename or '%(title)s [%(id)s]'}.%(ext)s"),
        "postprocessors": postprocessors,
        "logger": YdlLogAdapter(logger),
        "progress_hooks": [make_progress_hook(logger)],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
    }


def download(
    url: str, output_dir: Path, audio_format: str, logger: logging.Logger,
    filename: str | None = None,
) -> Path | None:
    """Baixa a URL. Devolve o caminho do arquivo final, ou None se falhou.

    filename fixa o nome (sem extensão); o padrão é "<título> [<id>]".
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    opts = build_ydl_opts(output_dir, audio_format, logger, filename)
    logger.info(f"Buscando informações: {url}")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        logger.error(f"Falha ao baixar {url}: {exc}")
        return None

    title = info.get("title", "?")
    abr = info.get("abr") or info.get("tbr")
    codec = info.get("acodec", "?")
    duration = info.get("duration_string", "?")
    source = f"~{abr:.0f} kbps" if abr else "?"
    logger.info(f"✅ {title}")
    if audio_format == "mp3":
        logger.info(f"   formato: mp3 320 kbps (convertido de {codec} {source}) | duração: {duration}")
    elif audio_format == "best":
        logger.info(f"   codec: {codec} | bitrate: {source} | duração: {duration}")
    else:
        logger.info(f"   formato: {audio_format} (convertido de {codec} {source}) | duração: {duration}")
    logger.info(f"   pasta: {output_dir}")
    return resolve_output_path(info, output_dir, logger, filename)


def resolve_output_path(
    info: dict, output_dir: Path, logger: logging.Logger, filename: str | None = None
) -> Path | None:
    """Descobre o arquivo que o yt-dlp acabou de escrever.

    Os postprocessors trocam a extensão depois do download, então o caminho
    do requested_downloads pode apontar para o arquivo pré-conversão. Tentamos
    o que o yt-dlp reporta e, se não existir, procuramos pelo ID do vídeo.
    """
    for entry in info.get("requested_downloads") or []:
        for key in ("filepath", "_filename", "filename"):
            candidate = entry.get(key)
            if candidate and Path(candidate).exists():
                return Path(candidate)

    video_id = info.get("id")
    pattern = f"{filename}.*" if filename else f"*[[]{video_id}[]]*"
    if filename or video_id:
        matches = sorted(
            (p for p in output_dir.glob(pattern) if p.is_file()
             and p.suffix.lower() not in {".webp", ".jpg", ".png", ".part"}),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]

    logger.warning("Não localizei o arquivo baixado para separar os stems.")
    return None


def cover_dir(name: str) -> Path:
    """Cria (se preciso) a pasta de um cover e devolve onde o áudio deve cair.

    Estrutura: ~/Music/Covers/<nome>/ com as quatro pastas do fluxo. O download e
    os stems vão para "song and stems/"; as outras ficam prontas para uso.
    """
    base = COVERS_DIR / name
    for sub in COVER_SUBDIRS:
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base / AUDIO_SUBDIR


def stems_dest(audio: Path, output_dir: Path) -> Path:
    """Pasta de stems de uma música.

    Dentro de um cover os stems ficam em "song and stems/stems/" — um cover tem
    uma música só, então não precisa do nível por faixa. Fora, agrupa por nome.
    """
    if output_dir.name == AUDIO_SUBDIR and output_dir.parent.parent == COVERS_DIR:
        return output_dir / "stems"
    return output_dir / "stems" / audio.stem


def separate_file(
    audio: Path,
    output_dir: Path,
    logger: logging.Logger,
    mode: str,
    device: str,
    fmt: str,
    shifts: int,
) -> bool:
    """Separa um arquivo de áudio em stems. Devolve True se deu certo."""
    import stems as stems_mod

    dest = stems_dest(audio, output_dir)
    logger.info(f"🎸 Separando instrumentos: {audio.name}")
    try:
        produced = stems_mod.separate(
            audio, dest, logger, mode=mode, device=device, fmt=fmt, shifts=shifts
        )
    except Exception as exc:
        logger.error(f"Falha ao separar {audio.name}: {exc}")
        logger.debug("detalhe:", exc_info=True)
        return False

    if not produced:
        logger.error(f"Nenhum stem produzido para {audio.name}.")
        return False

    logger.info(f"✅ {len(produced)} stems em {dest}")
    for name in sorted(produced):
        logger.info(f"   • {produced[name].name}")
    return True


# ------------------------------------------------------------ interactive ---

BANNER = """\033[1;35m
  ┌──────────────────────────────────────────────────┐
  │  ♪  yt-audio — YouTube → MP3 320 + stems 🎸      │
  └──────────────────────────────────────────────────┘\033[0m
  Cole um link do YouTube e aperte Enter.
  Comandos: \033[36mq\033[0m sair · \033[36mdir <pasta>\033[0m mudar saída · \033[36mfmt <mp3|best|m4a|opus|flac>\033[0m
            \033[36mstems on|off\033[0m separar instrumentos · \033[36mmode <full|6s|guitar>\033[0m
"""


def interactive(
    output_dir: Path,
    audio_format: str,
    logger: logging.Logger,
    stems: bool,
    stems_mode: str,
    device: str,
    stems_format: str,
    shifts: int,
) -> None:
    print(BANNER)
    while True:
        flag = f" +stems:{stems_mode}" if stems else ""
        try:
            line = input(
                f"\033[1m[{audio_format}{flag} → {output_dir}]\033[0m ♪ "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            logger.info("Até mais! 👋")
            return
        if not line:
            continue
        if line.lower() in {"q", "quit", "exit", "sair"}:
            logger.info("Até mais! 👋")
            return
        if line.lower().startswith("dir "):
            output_dir = Path(line[4:].strip()).expanduser()
            logger.info(f"Pasta de saída: {output_dir}")
            continue
        if line.lower().startswith("fmt "):
            audio_format = line[4:].strip().lower()
            logger.info(f"Formato: {audio_format}")
            continue
        if line.lower() in {"stems on", "stems"}:
            stems = True
            logger.info(f"Separação de instrumentos: ligada (modo {stems_mode})")
            continue
        if line.lower() == "stems off":
            stems = False
            logger.info("Separação de instrumentos: desligada")
            continue
        if line.lower().startswith("mode "):
            wanted = line[5:].strip().lower()
            if wanted in STEMS_MODES:
                stems_mode = wanted
                logger.info(f"Modo de stems: {stems_mode}")
            else:
                logger.warning(f"Modo inválido. Use: {', '.join(STEMS_MODES)}")
            continue
        if not line.startswith(("http://", "https://")):
            logger.warning("Isso não parece um link. Cole uma URL do YouTube.")
            continue

        audio = download(line, output_dir, audio_format, logger)
        if audio and stems:
            separate_file(
                audio, output_dir, logger, stems_mode, device, stems_format, shifts
            )


# ---------------------------------------------------------------- main ------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="yt-audio",
        description="Baixa áudio do YouTube na melhor qualidade e separa os instrumentos.",
    )
    parser.add_argument("urls", nargs="*", help="links do YouTube (vazio = modo interativo)")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"pasta de saída (padrão: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "-c", "--cover", metavar="NOME",
        help="monta ~/Music/Covers/NOME/ com as 4 pastas do fluxo",
    )
    parser.add_argument(
        "-f", "--format", default="mp3",
        choices=["mp3", "best", "m4a", "opus", "flac", "wav"],
        help="formato do áudio (padrão: mp3 320 kbps; 'best' mantém o codec original sem re-encode)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log detalhado no console")

    stems_group = parser.add_argument_group("separação de instrumentos")
    stems_group.add_argument(
        "-s", "--stems", action="store_true",
        help="separa os instrumentos depois de baixar",
    )
    stems_group.add_argument(
        "--stems-only", nargs="+", type=Path, metavar="ARQUIVO",
        help="separa arquivos de áudio locais (sem baixar nada)",
    )
    stems_group.add_argument(
        "--stems-mode", default="full", choices=STEMS_MODES,
        help="full: 6 stems + guitarra dedicada (padrão) | 6s: só htdemucs_6s | "
             "guitar: guitarra + base sem guitarra | "
             "noguitar: só a base, todos os instrumentos menos a guitarra",
    )
    stems_group.add_argument(
        "--no-guitar", dest="stems_mode", action="store_const", const="noguitar",
        help="atalho para --stems-mode noguitar",
    )
    stems_group.add_argument(
        "--stems-format", default="wav", choices=["wav", "flac", "mp3"],
        help="formato dos stems (padrão: wav 24-bit, sem perda)",
    )
    stems_group.add_argument(
        "--device", default="auto", choices=["auto", "mps", "cuda", "cpu"],
        help="onde rodar os modelos (padrão: auto — usa Metal/MPS no Apple Silicon)",
    )
    stems_group.add_argument(
        "--shifts", type=int, default=1, metavar="N",
        help="passes do demucs com deslocamento aleatório: mais qualidade, N vezes "
             "mais lento (padrão: 1)",
    )
    args = parser.parse_args()

    logger = setup_logging(args.verbose)
    output_dir = cover_dir(args.cover) if args.cover else args.output_dir.expanduser()
    if args.cover:
        logger.info(f"🎬 Cover: {output_dir.parent}")
    logger.debug(
        f"Args: urls={args.urls} output_dir={output_dir} format={args.format} "
        f"stems={args.stems} mode={args.stems_mode} device={args.device}"
    )

    # separar arquivos locais, sem baixar
    if args.stems_only:
        failures = 0
        for path in args.stems_only:
            path = path.expanduser()
            if not path.exists():
                logger.error(f"Arquivo não encontrado: {path}")
                failures += 1
                continue
            if not separate_file(
                path, output_dir, logger, args.stems_mode, args.device,
                args.stems_format, args.shifts,
            ):
                failures += 1
        return 1 if failures else 0

    if not args.urls:
        interactive(
            output_dir, args.format, logger, args.stems, args.stems_mode,
            args.device, args.stems_format, args.shifts,
        )
        return 0

    failures = 0
    for url in args.urls:
        audio = download(url, output_dir, args.format, logger)
        if audio is None:
            failures += 1
            continue
        if args.stems and not separate_file(
            audio, output_dir, logger, args.stems_mode, args.device,
            args.stems_format, args.shifts,
        ):
            failures += 1
    if failures:
        logger.error(f"{failures} de {len(args.urls)} itens falharam.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
