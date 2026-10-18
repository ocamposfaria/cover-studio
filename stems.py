"""
stems — separação de instrumentos (source separation) local e gratuita.

Dois modelos, complementares:

  htdemucs_6s (Meta/Demucs, MIT)
      Seis stems: drums, bass, vocals, guitar, piano, other.
      O autor do Demucs avisa: guitarra sai "okay", piano vaza bastante.
      É o único jeito gratuito de ter piano e guitarra separados num só passe.

  MelBand-Roformer guitar (becruily)
      Modelo dedicado só a guitarra, arquitetura mais nova (Roformer).
      Qualidade de guitarra bem superior ao htdemucs_6s. Só produz guitarra.

O modo "full" roda os dois: o set completo de seis stems vem do htdemucs_6s,
e a guitarra boa (Roformer) sobrescreve a fraca — a do 6s fica salva ao lado
como guitar_6s.wav para comparação.

Nota sobre chunking: o Roformer processa em janelas de 11s. Emendar janela
com janela direto gera um clique audível a cada emenda (medimos: salto 2.6x
maior que o transiente mais alto da música). Por isso usamos overlap-add com
janela de Hann e hop de 50% — derruba a emenda para 0.9x, ou seja, inaudível.
Custa ~57% mais tempo e vale cada segundo.
"""

from __future__ import annotations

import functools
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from console import progress

MODELS_DIR = Path(__file__).resolve().parent / "models"
GUITAR_CKPT = MODELS_DIR / "becruily_guitar.ckpt"
GUITAR_CONFIG = MODELS_DIR / "config_guitar_becruily.yaml"

GUITAR_REPO = "becruily/mel-band-roformer-guitar"
DEMUCS_MODEL = "htdemucs_6s"
SIX_STEMS = ("drums", "bass", "vocals", "guitar", "piano", "other")

# htdemucs_6s stems que somados formam a base "sem guitarra"
NO_GUITAR_PARTS = ("drums", "bass", "vocals", "piano", "other")


# ------------------------------------------------------------------ device ---

def pick_device(requested: str, logger: logging.Logger) -> str:
    """Escolhe mps (Metal) quando disponível, senão cpu.

    Contrariando muito material antigo, MPS funciona nesses modelos hoje e é
    ~1.8x mais rápido que CPU no Apple Silicon.
    """
    import torch

    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    logger.debug("Nem MPS nem CUDA disponíveis; usando CPU.")
    return "cpu"


# ------------------------------------------------------------------ demucs ---

def run_demucs(
    audio: Path,
    out_dir: Path,
    device: str,
    logger: logging.Logger,
    shifts: int = 1,
) -> dict[str, Path]:
    """Roda htdemucs_6s e devolve {nome_do_stem: caminho}.

    Chama via subprocess (python -m demucs) em vez de importar: o Demucs
    gerencia download de pesos, splitting e overlap internamente, e a CLI é
    a interface estável dele.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", DEMUCS_MODEL,
        "-d", device,
        "--out", str(out_dir),
        "--filename", "{stem}.{ext}",
        "--shifts", str(shifts),
        "--int24",                  # default do demucs é 16 bits; queremos 24
        "--clip-mode", "rescale",   # reescala em vez de estourar em 0 dBFS
        str(audio),
    ]
    logger.info(f"Separando 6 stems com {DEMUCS_MODEL} ({device})…")
    logger.debug("cmd: " + " ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError(
            "demucs falhou:\n" + "\n".join(tail[-12:] or ["(sem saída)"])
        )

    # demucs escreve em <out>/<modelo>/<stem>.wav
    produced = out_dir / DEMUCS_MODEL
    stems: dict[str, Path] = {}
    for name in SIX_STEMS:
        path = produced / f"{name}.wav"
        if path.exists():
            stems[name] = path
        else:
            logger.warning(f"stem esperado não apareceu: {name}")
    if not stems:
        raise RuntimeError(f"demucs não produziu stems em {produced}")
    return stems


# ----------------------------------------------------------------- roformer ---

def ensure_guitar_model(logger: logging.Logger) -> bool:
    """Garante checkpoint + config do Roformer de guitarra localmente."""
    if GUITAR_CKPT.exists() and GUITAR_CONFIG.exists():
        return True

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Baixando modelo de guitarra ({GUITAR_REPO}, ~45 MB)…")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub não disponível para baixar o modelo.")
        return False

    try:
        for remote, local in (
            ("becruily_guitar.ckpt", GUITAR_CKPT),
            ("config_guitar_becruily.yaml", GUITAR_CONFIG),
        ):
            if local.exists():
                continue
            got = hf_hub_download(repo_id=GUITAR_REPO, filename=remote)
            shutil.copyfile(got, local)
        return True
    except Exception as exc:  # rede, repo movido, etc.
        logger.error(f"Não consegui baixar o modelo de guitarra: {exc}")
        return False


def read_audio_44k(
    audio: Path, target_sr: int, logger: logging.Logger
) -> tuple["np.ndarray", int]:  # noqa: F821
    """Lê o áudio garantindo `target_sr`, resampleando via ffmpeg se preciso.

    O YouTube serve Opus a 48 kHz, mas o Roformer de guitarra foi treinado em
    44.1 kHz — sem resample ele nem roda.

    Preferimos o soxr em precisão 28 bits (transparente), mas nem todo build
    de ffmpeg traz libsoxr (o do Homebrew, por exemplo, não traz). Nesse caso
    caímos no resampler nativo com filtro longo e fase precisa, que também é
    inaudível para este fim. Sempre em float32, para não perder bits no meio.
    """
    import numpy as np  # noqa: F401  (usado pelo tipo de retorno)
    import soundfile as sf

    info = sf.info(str(audio))
    if info.samplerate == target_sr:
        data, sr = sf.read(str(audio), dtype="float32", always_2d=True)
        return data, sr

    logger.info(
        f"Resampleando {info.samplerate} → {target_sr} Hz para o modelo de guitarra…"
    )
    tmp = audio.parent / f".{audio.stem}.{target_sr}.wav"
    filters = (
        f"aresample={target_sr}:resampler=soxr:precision=28",
        f"aresample={target_sr}:filter_size=256:phase_shift=10",
    )

    last_err = ""
    for audio_filter in filters:
        proc = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(audio),
                "-af", audio_filter,
                "-ac", "2", "-c:a", "pcm_f32le",
                str(tmp),
            ],
            capture_output=True, text=True,
        )
        if proc.returncode == 0:
            if "soxr" not in audio_filter:
                logger.debug("libsoxr indisponível; usei o resampler nativo do ffmpeg.")
            try:
                return sf.read(str(tmp), dtype="float32", always_2d=True)
            finally:
                tmp.unlink(missing_ok=True)
        last_err = (proc.stderr or "").strip()

    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"ffmpeg falhou ao resamplear: {last_err[:300]}")


def _load_guitar_model(device: str):
    """Carrega o MelBand-Roformer de guitarra.

    Dois remendos necessários, ambos por bugs de terceiros:

    1. mlp_expansion_factor — o config do becruily usa 1, mas o
       get_model_from_config do melband-roformer-infer descarta a chave
       silenciosamente e MaskEstimator cai no default 4. Resultado: um mask
       estimator de 1024 dims contra um checkpoint de 256 → ~180 erros de
       size mismatch. Injetamos o valor via partialmethod.

    2. yaml.unsafe_load — o config traz !!python/tuple, que safe_load recusa.
       O arquivo é nosso (vem do models/), então unsafe_load é aceitável aqui.
    """
    import torch
    import yaml
    from ml_collections import ConfigDict
    from mel_band_roformer import mel_band_roformer as mb

    mb.MaskEstimator.__init__ = functools.partialmethod(
        mb.MaskEstimator.__init__, mlp_expansion_factor=1
    )

    with open(GUITAR_CONFIG) as fh:
        cfg = ConfigDict(yaml.unsafe_load(fh))

    model_cfg = {
        k: v for k, v in dict(cfg.model).items() if k != "mlp_expansion_factor"
    }
    model = mb.MelBandRoformer(**model_cfg)

    state = torch.load(GUITAR_CKPT, map_location="cpu", weights_only=True)
    # strict=True de propósito: se o remendo do mlp_expansion_factor parar de
    # funcionar, queremos estourar aqui e não gerar áudio silenciosamente ruim.
    model.load_state_dict(state.get("state_dict", state), strict=True)
    return model.to(device).eval(), cfg


def separate_guitar(
    audio: Path,
    out_path: Path,
    device: str,
    logger: logging.Logger,
    overlap: float = 0.5,
) -> Path:
    """Extrai a guitarra com o Roformer dedicado, via overlap-add.

    overlap=0.5 (hop de meia janela) com janela de Hann. Ver a nota sobre
    chunking no topo do módulo: sem isso, clique audível a cada 11 s.
    """
    import numpy as np
    import soundfile as sf
    import torch

    model, cfg = _load_guitar_model(device)
    sr_model = int(cfg.audio.sample_rate)
    chunk = int(cfg.audio.chunk_size)

    # O YouTube entrega Opus a 48 kHz e o modelo é treinado em 44.1 kHz;
    # resample é obrigatório, não opcional (ver read_audio_44k).
    data, sr = read_audio_44k(audio, sr_model, logger)
    if data.shape[1] == 1:  # mono → o modelo é estéreo
        data = np.repeat(data, 2, axis=1)

    x = torch.from_numpy(data.T)[None].to(device)
    n = x.shape[-1]

    hop = max(1, int(chunk * (1.0 - overlap)))
    window = torch.hann_window(chunk, periodic=True, device=device)
    acc = torch.zeros_like(x)
    weight = torch.zeros(n, device=device)

    starts = list(range(0, max(n - chunk, 0) + hop, hop)) or [0]
    logger.info(
        f"Isolando guitarra com MelBand-Roformer ({device}, "
        f"{len(starts)} janelas, overlap {overlap:.0%})…"
    )

    with torch.no_grad():
        for i, start in enumerate(starts, 1):
            seg = x[..., start:start + chunk]
            length = seg.shape[-1]
            if length == 0:
                continue
            pad = chunk - length
            if pad > 0:
                seg = torch.nn.functional.pad(seg, (0, pad))

            y = model(seg)
            if y.dim() == 4:  # (batch, stems, ch, t) → único stem
                y = y[:, 0]
            y = y[..., :length]

            w = window[:length]
            acc[..., start:start + length] += y * w
            weight[start:start + length] += w

            progress("guitarra", i / len(starts), f"janela {i}/{len(starts)}",
                     final=i == len(starts))

    guitar = (acc / weight.clamp(min=1e-8))[0].cpu().numpy().T
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), guitar, sr, subtype="PCM_24")
    return out_path


# -------------------------------------------------------------- mixdown ------

# ~0.3 dB de folga. Música masterizada chega exatamente em 1.0, e somar ou
# subtrair stems empurra além disso; sem folga o WAV satura no limite.
HEADROOM = 0.967


def _fit_headroom(signal, label: str, logger: logging.Logger):
    """Reescala o sinal se ele encostar em 0 dBFS, preservando a dinâmica."""
    import numpy as np

    peak = float(np.abs(signal).max())
    if peak > HEADROOM:
        logger.debug(f"{label}: pico {peak:.3f} → reescalando para {HEADROOM}")
        return signal * (HEADROOM / peak)
    return signal

def mix_without_guitar(
    stems: dict[str, Path],
    out_path: Path,
    logger: logging.Logger,
) -> Path | None:
    """Base sem guitarra somando os cinco stems não-guitarra do htdemucs_6s.

    Usado só quando não há guitarra do Roformer. Com ela, `_subtract_guitar`
    faz melhor: preserva tudo que o htdemucs_6s tenha jogado fora.
    """
    import numpy as np
    import soundfile as sf

    parts = [stems[name] for name in NO_GUITAR_PARTS if name in stems]
    if not parts:
        return None

    total = None
    sr = None
    for path in parts:
        data, this_sr = sf.read(str(path), dtype="float32", always_2d=True)
        sr = this_sr if sr is None else sr
        if total is None:
            total = data
            continue
        n = min(len(total), len(data))  # trunca ao menor; stems podem diferir 1-2 samples
        total = total[:n] + data[:n]

    if total is None:
        return None

    total = _fit_headroom(total, "base sem guitarra", logger)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), total, sr, subtype="PCM_24")
    return out_path


# ------------------------------------------------------------------ encode ---

def encode_stem(src: Path, fmt: str, logger: logging.Logger) -> Path:
    """Converte um stem WAV para flac/mp3 quando pedido. wav = no-op."""
    if fmt == "wav":
        return src

    dst = src.with_suffix(".flac" if fmt == "flac" else ".mp3")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src)]
    if fmt == "flac":
        cmd += ["-c:a", "flac", "-compression_level", "8"]
    else:
        cmd += ["-c:a", "libmp3lame", "-b:a", "320k"]
    cmd.append(str(dst))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning(f"falha convertendo {src.name} para {fmt}; mantendo wav")
        return src
    src.unlink(missing_ok=True)
    return dst


# ----------------------------------------------------------------- driver ----

def separate(
    audio: Path,
    dest: Path,
    logger: logging.Logger,
    mode: str = "full",
    device: str = "auto",
    fmt: str = "wav",
    shifts: int = 1,
) -> dict[str, Path]:
    """Separa `audio` em stems dentro de `dest`.

    mode:
      full      htdemucs_6s (6 stems) + guitarra do Roformer por cima
      6s        só htdemucs_6s
      guitar    guitarra do Roformer + base sem guitarra
      noguitar  só a base sem guitarra (todos os outros instrumentos juntos)
    """
    device = pick_device(device, logger)
    dest.mkdir(parents=True, exist_ok=True)
    work = dest / ".work"
    results: dict[str, Path] = {}

    try:
        demucs_stems: dict[str, Path] = {}
        if mode in {"full", "6s"}:
            demucs_stems = run_demucs(audio, work, device, logger, shifts)
            for name, path in demucs_stems.items():
                final = dest / f"{name}.wav"
                shutil.move(str(path), final)
                demucs_stems[name] = final
                results[name] = final

        has_roformer_guitar = False
        if mode in {"full", "guitar", "noguitar"} and ensure_guitar_model(logger):
            # no modo full a guitarra do 6s vira guitar_6s para comparação
            if "guitar" in results:
                legacy = dest / "guitar_6s.wav"
                shutil.move(str(results["guitar"]), legacy)
                results["guitar_6s"] = legacy

            try:
                results["guitar"] = separate_guitar(
                    audio, dest / "guitar.wav", device, logger
                )
                has_roformer_guitar = True
            except Exception as exc:
                logger.error(f"Roformer de guitarra falhou: {exc}")
                # devolve a guitarra do 6s ao lugar, se existia
                if "guitar_6s" in results:
                    shutil.move(str(results.pop("guitar_6s")), dest / "guitar.wav")
                    results["guitar"] = dest / "guitar.wav"
                    logger.warning("Usando a guitarra do htdemucs_6s (qualidade menor).")

        # Base sem guitarra. Preferimos (mix - guitarra do Roformer): preserva
        # sintetizador, teclado e tudo que o htdemucs_6s tenha borrado. Só
        # caímos na soma dos stems quando não houver guitarra do Roformer.
        base = None
        if has_roformer_guitar:
            base = _subtract_guitar(audio, results["guitar"], dest, logger)
        if base is None and demucs_stems:
            base = mix_without_guitar(demucs_stems, dest / "no_guitar.wav", logger)
        if base:
            results["no_guitar"] = base

        # noguitar: a guitarra era só o meio para obter a base; não é entregável
        if mode == "noguitar" and "guitar" in results:
            results.pop("guitar").unlink(missing_ok=True)

        if fmt != "wav":
            logger.info(f"Convertendo stems para {fmt}…")
            results = {k: encode_stem(v, fmt, logger) for k, v in results.items()}

    finally:
        shutil.rmtree(work, ignore_errors=True)

    return results


def _subtract_guitar(
    audio: Path, guitar: Path | None, dest: Path, logger: logging.Logger
) -> Path | None:
    """base = mix - guitarra. Preserva synth, teclado, tudo que não é guitarra."""
    if guitar is None or not guitar.exists():
        return None

    import numpy as np
    import soundfile as sf

    # a guitarra sai a 44.1 kHz; o mix pode estar a 48 kHz (Opus do YouTube).
    # Alinhamos o mix à taxa da guitarra, senão a subtração sai sem sentido.
    gtr, sr = sf.read(str(guitar), dtype="float32", always_2d=True)
    mix, _ = read_audio_44k(audio, sr, logger)
    if mix.shape[1] == 1:
        mix = np.repeat(mix, 2, axis=1)
    length = min(len(mix), len(gtr))
    base = mix[:length] - gtr[:length]

    base = _fit_headroom(base, "base sem guitarra", logger)

    out = dest / "no_guitar.wav"
    sf.write(str(out), base, sr, subtype="PCM_24")
    return out
