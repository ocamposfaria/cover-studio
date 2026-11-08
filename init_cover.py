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
init_cover — começa um cover do zero.

Uso:
    ./init_cover.py                                  # pergunta tudo
    ./init_cover.py "nirvana" "lounge act"           # artista e música
    ./init_cover.py "nirvana" "lounge act" <url>     # já com o link

Pergunta qual é o cover, acha a música no YouTube (ou usa o link que você
colar), e monta ~/Music/Covers/<artista> - <música>/:

    song and stems/<música>.mp3 + stems/     baixado e separado (modo full)
    garageband project/Projeto.band          cópia de templates/cover.band
    exported mixes/  final video/            vazias, prontas para uso

No fim abre o projeto no GarageBand e mostra os stems no Finder. O GarageBand
não tem API para inserir faixas, então arrastar os stems é com você.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yt_dlp

import yt_audio
from yt_audio import AUDIO_SUBDIR, COVERS_DIR

TEMPLATE = yt_audio.SCRIPT_DIR / "templates" / "cover.band"
BAND_SUBDIR = "garageband project"
BAND_NAME = "Projeto.band"  # todo cover tem o mesmo nome de projeto; a pasta diz qual é
SEARCH_RESULTS = 5


def ask(prompt: str, default: str | None = None) -> str:
    hint = f" [{default}]" if default else ""
    while True:
        try:
            answer = input(f"\033[1m{prompt}{hint}:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(130)
        if answer or default is not None:
            return answer or default


def search_youtube(query: str) -> list[dict]:
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{SEARCH_RESULTS}:{query}", download=False)
    return [e for e in info.get("entries") or [] if e]


def pick_url(artist: str, song: str) -> str:
    """Busca no YouTube e deixa escolher; também aceita um link colado."""
    query = f"{artist} - {song}"
    while True:
        print(f"\n🔎 Buscando no YouTube: {query}")
        results = search_youtube(query)
        for i, e in enumerate(results, 1):
            secs = int(e.get("duration") or 0)
            dur = f"{secs // 60}:{secs % 60:02d}" if secs else "?"
            channel = e.get("channel") or e.get("uploader") or "?"
            print(f"  \033[36m{i}\033[0m  {e.get('title')}  \033[2m— {channel} · {dur}\033[0m")
        answer = ask("\nNúmero, link do YouTube, ou outra busca", "1")
        if answer.startswith(("http://", "https://")):
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(results):
            e = results[int(answer) - 1]
            return e.get("url") or f"https://www.youtube.com/watch?v={e['id']}"
        query = answer


def existing_audio(audio_dir: Path) -> Path | None:
    for p in sorted(audio_dir.glob("*")):
        if p.is_file() and p.suffix.lower() in {".mp3", ".m4a", ".wav", ".flac", ".opus"}:
            return p
    return None


def make_band(base: Path, logger) -> Path | None:
    """Copia o template do GarageBand para o cover, se ainda não houver projeto."""
    band_dir = base / BAND_SUBDIR
    existing = sorted(band_dir.glob("*.band"))
    if existing:
        logger.info(f"🎛  Projeto do GarageBand já existe: {existing[0].name}")
        return existing[0]
    if not TEMPLATE.exists():
        logger.warning(f"Sem template em {TEMPLATE}; crie o projeto no GarageBand à mão.")
        return None
    band = band_dir / BAND_NAME
    shutil.copytree(TEMPLATE, band)
    logger.info(f"🎛  Projeto do GarageBand: {band}")
    return band


def main() -> int:
    parser = argparse.ArgumentParser(prog="init_cover", description="Começa um cover do zero.")
    parser.add_argument("artist", nargs="?", help="artista (vazio = pergunta)")
    parser.add_argument("song", nargs="?", help="música (vazio = pergunta)")
    parser.add_argument("url", nargs="?", help="link do YouTube (vazio = busca)")
    parser.add_argument(
        "--stems-mode", default="full", choices=yt_audio.STEMS_MODES,
        help="modo de separação (padrão: full)",
    )
    parser.add_argument("--no-open", action="store_true", help="não abre GarageBand nem Finder no fim")
    parser.add_argument("-v", "--verbose", action="store_true", help="log detalhado no console")
    args = parser.parse_args()

    artist = (args.artist or ask("Artista")).strip().lower()
    song = (args.song or ask("Música")).strip().lower()
    name = f"{artist} - {song}"

    logger = yt_audio.setup_logging(args.verbose)
    base = COVERS_DIR / name
    if base.exists():
        logger.info(f"Cover já existe, completando o que falta: {base}")
    audio_dir = yt_audio.cover_dir(name)
    logger.info(f"🎬 Cover: {base}")

    audio = existing_audio(audio_dir)
    if audio:
        logger.info(f"🎵 Música já baixada: {audio.name}")
    else:
        url = args.url or pick_url(artist, song)
        audio = yt_audio.download(url, audio_dir, "mp3", logger, filename=song)
        if audio is None:
            return 1

    stems = audio_dir / "stems"
    if stems.exists() and any(stems.glob("no_guitar.*")):
        logger.info("🎸 Stems já separados.")
    elif not yt_audio.separate_file(audio, audio_dir, logger, args.stems_mode, "auto", "wav", 1):
        return 1

    band = make_band(base, logger)

    logger.info(f"✅ Pronto: {base}")
    if not args.no_open:
        if band:
            subprocess.run(["open", str(band)], check=False)
        reveal = next(iter(sorted(stems.glob("no_guitar.*"))), stems)
        subprocess.run(["open", "-R", str(reveal)], check=False)
        logger.info("Arraste os stems do Finder para o GarageBand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
