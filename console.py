"""Saída no terminal: formatter colorido do log e barra de progresso.

Tudo vai para o stderr, o mesmo stream do log; assim a barra e as linhas de
log não se embaralham. Fora de um terminal (pipe, arquivo) não há cor nem barra.
"""

from __future__ import annotations

import logging
import sys

LEVELS = {
    logging.DEBUG: ("dbg", "2"),        # dim
    logging.INFO: ("info", "36"),       # cyan
    logging.WARNING: ("aviso", "33"),   # yellow
    logging.ERROR: ("erro", "31"),      # red
    logging.CRITICAL: ("erro", "1;31"), # bold red
}
BAR_WIDTH = 28


def is_tty() -> bool:
    return sys.stderr.isatty()


def paint(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if is_tty() else text


class ColorFormatter(logging.Formatter):
    """`HH:MM:SS info  mensagem`, com aviso/erro coloridos e ✅ em verde."""

    def format(self, record: logging.LogRecord) -> str:
        label, color = LEVELS.get(record.levelno, ("?", "0"))
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        if record.levelno >= logging.WARNING:
            msg = paint(color, msg)
        elif record.levelno == logging.DEBUG:
            msg = paint("2", msg)
        elif msg.startswith("✅"):
            msg = paint("1;32", msg)
        stamp = paint("2", self.formatTime(record, "%H:%M:%S"))
        return f"{stamp} {paint(color, f'{label:<5}')} {msg}"


def progress(label: str, fraction: float, detail: str = "", final: bool = False) -> None:
    """Desenha (ou redesenha) uma barra na linha atual; final=True fecha a linha."""
    if not is_tty():
        return
    fraction = min(max(fraction, 0.0), 1.0)
    filled = round(BAR_WIDTH * fraction)
    bar = paint("35", "█" * filled) + paint("2", "░" * (BAR_WIDTH - filled))
    line = f"\r\033[K         {paint('35', label)} {bar} {fraction:4.0%}"
    if detail:
        line += "  " + paint("2", detail)
    sys.stderr.write(line + ("\n" if final else ""))
    sys.stderr.flush()
