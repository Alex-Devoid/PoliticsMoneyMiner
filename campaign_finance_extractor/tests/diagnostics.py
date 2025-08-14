# diagnostics.py  ------------------------------------------------------------
# One‑file live monitor for Sedgwick OCR work:
#   • main Python process   (where the scraper runs)
#   • every thread in that process (shows TrOCR activity)
#   • every live `tesseract` child process
# Refreshes every `interval` seconds (Ctrl‑C to stop).
# ---------------------------------------------------------------------------
from __future__ import annotations

import os
import shutil
import time
from typing import List

import psutil

TESS_PATTERN   = "tesseract"                  # short‑lived subprocesses
TROCR_PATTERN  = "trocr-small-handwritten"    # appears in thread name
PY_PATTERN     = "SedgwickExpenseScraper"     # import path part

# ── helper ---------------------------------------------------------------
def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _tail(cmd: List[str], n: int = 4) -> str:
    return " ".join(cmd[-n:]) if cmd else ""


# ── main monitor ---------------------------------------------------------
def watch(interval: float = 1.0, repeat: int = 0) -> None:
    """
    Refresh the table every `interval` s.  If `repeat` > 0, exit after that
    many refreshes.
    """
    me = psutil.Process()                         # our own watcher
    parent_py: psutil.Process | None = None       # will hold scraper pid

    # prime CPU counters so first numbers are meaningful
    for p in psutil.process_iter():
        p.cpu_percent(None)

    while True:
        # ── locate the scraper process once ---------------------------------
        if parent_py is None:
            for p in psutil.process_iter(["pid", "cmdline"]):
                try:
                    if any(PY_PATTERN in part for part in (p.info["cmdline"] or [])):
                        parent_py = p
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        # ── gather rows -----------------------------------------------------
        rows: list[tuple[str, float, float, str]] = []

        # * tesseract children *
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                if TESS_PATTERN in " ".join(p.info["cmdline"] or []):
                    rows.append((str(p.pid),
                                 p.cpu_percent(None),
                                 p.memory_percent(),
                                 _tail(p.info["cmdline"])))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # * main scraper + its threads *
        if parent_py and parent_py.is_running():
            try:
                rows.append((str(parent_py.pid),
                             parent_py.cpu_percent(None),
                             parent_py.memory_percent(),
                             _tail(parent_py.cmdline())))
                for th in parent_py.threads():
                    tid = th.id
                    name = f"(thr {tid})"
                    cpu  = th.user_time + th.system_time
                    # mark TrOCR threads if their name contains the pattern
                    try:
                        tname = psutil.Process(tid).name()
                    except Exception:
                        tname = ""
                    tag = "  ←TrOCR" if TROCR_PATTERN in tname else ""
                    rows.append((name, cpu * 100, 0.0, tag))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                parent_py = None   # will re‑discover next cycle

        # ── render ----------------------------------------------------------
        _clear()
        print(f"{'PID/Thread':>10}  {'CPU%':>6}  {'MEM%':>6}  CMD/Tag")
        print("─" * shutil.get_terminal_size((80, 20)).columns)
        if not rows:
            print("(no matching activity yet)")
        else:
            for pid, cpu, mem, tail in rows:
                print(f"{pid:>10}  {cpu:6.1f}  {mem:6.1f}  {tail}")

        # ── exit / sleep ----------------------------------------------------
        if repeat:
            repeat -= 1
            if repeat <= 0:
                break
        time.sleep(interval)


# -------------------------------------------------------------------------
if __name__ == "__main__":
    watch()
