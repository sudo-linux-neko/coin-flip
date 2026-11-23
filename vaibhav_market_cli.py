# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 Vaibhav — All rights reserved.
# Author: Vaibhav  # noqa: ERA001
# Final authority / power over this project: Vaibhav.
# License intent: GPL-v3-only by default (NEVER MIT). If this code is pushed
# to GitHub or made public in any way, the publisher MUST ask Vaibhav which
# license to publish under. GPL-v3-only is the default choice unless Vaibhav
# explicitly approves another copyleft license more appropriate to the project.
#
# vaibhav_market_cli.py — hyper-optimized anime market dashboard for
#   Debian 13 (WSL2) / i5-1335U (12 threads, AVX2+FMA+AVX_VNNI) / 7.6 GiB RAM
# Data: 100% FREE ONLY — OpenBB Platform (yfinance provider, no key) ->
#       yfinance direct -> Stooq CSV. No paid keys required.
# Perf: ThreadPoolExecutor(12) == CPU threads, NumPy vectorized indicators
#       (OpenBLAS Haswell AVX2), TTL caches, lazy heavy imports, time_ns,
#       raw UDP NTP (median of N servers), single Rich Live redraw.
# Run: source ~/ml_env/bin/activate && python vaibhav_market_cli.py [--help]
"""Anime market CLI by Vaibhav. Free data only. Max perf on i5-1335U + AVX2."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as _dt
import math
import os
import select as _select
import socket
import struct
import subprocess
import sys
import termios as _termios
import time
import tty as _tty
from functools import lru_cache
from pathlib import Path
from typing import Any, Self

# ---- max out threaded math libs for THIS machine (12 threads, AVX2) ----
os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")

import numpy as np

# Rich is preinstalled in ~/ml_env — no new installs, no system breakage.
from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    import pyfiglet
    _HAS_FIGLET = True
except Exception:
    _HAS_FIGLET = False

console = Console()

# ---------------------------------------------------------------- presets

DEFAULT_WATCH = [
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN",
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
    "^GSPC", "^NSEI", "BTC-USD",
]
INDICES = ["^GSPC", "^NDX", "^NSEI", "^BSESN", "BTC-USD"]

# Named watchlists: usa / india / crypto / all + your custom ones.
WATCHLISTS = {
    "mix": DEFAULT_WATCH,
    "usa": ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL",
            "AMD", "NFLX", "JPM", "^GSPC", "^NDX"],
    "india": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
              "ICICIBANK.NS", "SBIN.NS", "TATAMOTORS.NS", "ITC.NS",
              "^NSEI", "^BSESN"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"],
    "all": sorted(
        {*DEFAULT_WATCH, "META", "GOOGL", "AMD", "NFLX",
         "HDFCBANK.NS", "ICICIBANK.NS", "ETH-USD"},
    ),
}
WATCH_FILE = Path("~/.vaibhav_dojo_watch.json").expanduser()


def load_custom_lists() -> dict[str, list[str]]:
    """Load custom watchlists saved on disk."""
    try:
        import json as _j
        if WATCH_FILE.exists():
            with WATCH_FILE.open() as f:
                d = _j.load(f)
            if isinstance(d, dict):
                return {k: v for k, v in d.items() if isinstance(v, list) and v}
    except Exception:
        pass
    return {}


def save_custom_list(name: str, symbols: list[str]) -> None:
    """Save a named watchlist to disk."""
    import json as _j
    try:
        d = load_custom_lists()
        d[name] = symbols
        with WATCH_FILE.open("w") as f:
            _j.dump(d, f, indent=1)
    except Exception as e:
        console.print(f"[red]could not save list ({e})[/]")

THEMES = {
    "sakura": {"c1": (255, 122, 179), "c2": (167, 139, 250), "c3": (103, 232, 249)},
    "tokyo": {"c1": (34, 211, 238), "c2": (232, 121, 249), "c3": (251, 191, 36)},
    "abyss": {"c1": (96, 165, 250), "c2": (52, 211, 153), "c3": (196, 181, 253)},
    # real Bloomberg identity: black field, amber base, green up / red down
    "bloomberg": {"c1": (255, 176, 0), "c2": (255, 176, 0), "c3": (255, 220, 130)},
}

SAKURA_ART = r"""
      🌸      ･ﾟ✧      🌙         ･ﾟ      🌸
   ───────────── ⛩️  TORII · SAKURA NIGHT  ─────────────
        ∧＿∧   ╭──────────────────────────╮
       ( ｡•‿•｡) │  桜  NEO-TOKYO MARKET   │
       / つ🍡  │  DOJO  ·  精度  ·  夢    │
      しーＪ    ╰──────────────────────────╯
   ∵ ｡ ∴ ｡ ∵ 🌸 ｡ ∴ 🌸 ｡ ∵ ∴ ｡ ∵ ｡ 🌸 ∴
      ─────  富士山  ·  月  ·  桜吹雪  ─────
""".strip("\n")

TOKYO_ART = r"""
   ｡ﾟ☁︎｡  ────  NEON TOKYO  ────  ｡ﾟ☁︎｡
   ┌─┐ ┌─┐ ┌─┐  東京  24:00 霓虹  ┌─┐ ┌─┐ ┌─┐
   │█│ │█│ │█│  ▓▓  MARKET GRID  ▓▓  │█│ │█│ │█│
   └─┘ └─┘ └─┘  ⚡ 精度 ⚡ 速度 ⚡ 夢 └─┘ └─┘ └─┘
   ──╰──╯──╰──╯──  電車  ·  雨  ·  光  ──╰──╯──
""".strip("\n")

ABYSS_ART = r"""
      ✦  ･ﾟ  ───  STAR ABYSS  ───  ﾟ･  ✦
         ／＼    . * .  🌌  . * .    ／＼
        /  ／\  ◢  DATA CONSTELLATION ◣  /＼  \
        ╲╱    ╲╱  ✧ 株 ✧ 星 ✧ 夢 ✧  ╲╱    ╲╱
      ─────  深海  ·  星空  ·  静寂  ─────
""".strip("\n")

ARTS = {"sakura": SAKURA_ART, "tokyo": TOKYO_ART, "abyss": ABYSS_ART}

# ---------------------------------------------------------------- utils

def _lerp(a: tuple, b: tuple, t: float) -> tuple:
    return (int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))


@lru_cache(maxsize=64)
def gradient_title(text: str, theme: str = "sakura") -> Text:
    """Per-char RGB gradient. Precomputed + cached: O(n) once, O(1) after."""
    th = THEMES.get(theme, THEMES["sakura"])
    c1, c2, c3 = th["c1"], th["c2"], th["c3"]
    t = Text()
    n = max(1, len(text))
    for i, ch in enumerate(text):
        p = i / max(1, n - 1)
        c = _lerp(c1, c2, p * 2) if p < 0.5 else _lerp(c2, c3, (p - 0.5) * 2)
        t.append(ch, style=f"bold rgb({c[0]},{c[1]},{c[2]})")
    return t


@lru_cache(maxsize=8)
def gradient_art(theme: str = "sakura") -> Text:
    """Anime banner with vertical pink->violet->cyan gradient. Cached."""
    art = ARTS.get(theme, SAKURA_ART)
    th = THEMES.get(theme, THEMES["sakura"])
    c1, c2, c3 = th["c1"], th["c2"], th["c3"]
    lines = art.splitlines()
    t = Text()
    for li, line in enumerate(lines):
        p = li / max(1, len(lines) - 1)
        c = _lerp(c1, c2, p * 2) if p < 0.5 else _lerp(c2, c3, (p - 0.5) * 2)
        t.append(line + "\n", style=f"rgb({c[0]},{c[1]},{c[2]})")
    return t


def big_title(text: str, theme: str) -> Text:
    """Render a big banner title for the given theme."""
    if _HAS_FIGLET:
        try:
            fig = pyfiglet.figlet_format(text, font="slant")
            return gradient_art_cached_fig(fig, theme)
        except Exception:
            pass
    return gradient_title(text, theme)


@lru_cache(maxsize=8)
def gradient_art_cached_fig(fig: str, theme: str) -> Text:
    """Render cached figlet art with a vertical gradient."""
    th = THEMES.get(theme, THEMES["sakura"])
    c1, c2, c3 = th["c1"], th["c2"], th["c3"]
    lines = fig.splitlines()
    t = Text()
    for li, line in enumerate(lines):
        p = li / max(1, len(lines) - 1)
        c = _lerp(c1, c2, p * 2) if p < 0.5 else _lerp(c2, c3, (p - 0.5) * 2)
        t.append(line + "\n", style=f"bold rgb({c[0]},{c[1]},{c[2]})")
    return t


# ------------------------------------------------------- precise time
# Most accurate time available in userspace on this box:
#   CLOCK_REALTIME via time.time_ns() (1ns res, verified) +
#   NTP median offset from pool (raw UDP, parallel) +
#   timedatectl sync flag (WSL2: Hyper-V host clock).

NTP_SERVERS = ("pool.ntp.org", "time.google.com", "time.cloudflare.com")
_NTP_CACHE = {"t": 0.0, "offset": 0.0, "rtt": 0.0}
NTP_TTL = 300.0


def _ntp_query(host: str, timeout: float = 1.6) -> tuple[float, float] | None:
    """Single SNTP query. Returns (offset_sec, rtt_sec) or None. No deps."""
    try:
        addr = socket.gethostbyname(host)
        pkt = b"\x1b" + 47 * b"\0"
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(timeout)
        t0 = time.time()  # local send time (Unix)
        p1 = time.perf_counter()
        s.sendto(pkt, (addr, 123))
        data, _ = s.recvfrom(1024)
        p4 = time.perf_counter()
        t3 = time.time()  # local recv time (Unix)
        s.close()
        if len(data) < 48:
            return None
        tx_s, tx_f = struct.unpack("!II", data[40:48])
        tx = tx_s + tx_f / 2**32 - 2208988800  # NTP->Unix
        # offset = server_time - local_time, RTT-halved (standard SNTP)
        off = tx - (t0 + t3) / 2.0
        return (off, p4 - p1)
    except Exception:
        return None


def ntp_median_offset() -> tuple[float, float]:
    """Parallel NTP, median offset. Cached 5 min. Never blocks UI >2s."""
    now = time.monotonic()
    if now - _NTP_CACHE["t"] < NTP_TTL:
        return _NTP_CACHE["offset"], _NTP_CACHE["rtt"]
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(_ntp_query, h) for h in NTP_SERVERS]
        res = [f.result() for f in futs]
    vals = [r for r in res if r]
    if vals:
        offs = sorted(v[0] for v in vals)
        rtts = sorted(v[1] for v in vals)
        off, rtt = offs[len(offs) // 2], rtts[len(rtts) // 2]
    else:
        off, rtt = 0.0, -1.0
    _NTP_CACHE.update(t=now, offset=off, rtt=rtt)
    return off, rtt


def precise_time() -> dict[str, Any]:
    """Bundle of the most-accurate clocks on this machine."""
    ns = time.time_ns()  # CLOCK_REALTIME, 1ns res
    off, rtt = ntp_median_offset()
    corrected = ns / 1e9 + off
    try:
        out = subprocess.run(
            ["/usr/bin/timedatectl", "status"],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        ).stdout
        sync = "yes" if "System clock synchronized: yes" in out else (
            "no" if "synchronized" in out else "n/a (WSL2 host clock)")
    except Exception:
        sync = "n/a"
    tz = time.tzname
    loc = _dt.datetime.fromtimestamp(ns / 1e9).astimezone()
    utc = _dt.datetime.fromtimestamp(ns / 1e9, tz=_dt.UTC)
    corr_dt = _dt.datetime.fromtimestamp(corrected, tz=_dt.UTC)
    return {
        "ns": ns, "local": loc, "utc": utc, "corrected_utc": corr_dt,
        "offset_ms": off * 1000.0, "rtt_ms": rtt * 1000.0,
        "sync": sync, "tz": tz,
        "mono": time.monotonic(), "perf": time.perf_counter(),
    }


def time_panel(theme: str, mode: str = "ms") -> Panel:
    """Mode ms (default): 12h clock, 3-decimal ms. mode max: full ns detail."""
    _ = theme  # reserved for future theming; time layout is theme-agnostic
    p = precise_time()
    tbl = Table(show_header=False, box=None, pad_edge=False, expand=True)
    tbl.add_column("k", style="bold cyan", width=13)
    tbl.add_column("v", style="white")
    if mode == "max":
        tbl.add_row("Local", p["local"].strftime("%I:%M:%S.%f %p %Z"))
        tbl.add_row("UTC", p["utc"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3])
        tbl.add_row("time_ns", f"{p['ns']:,} ns")
        tbl.add_row("NTP-fix", p["corrected_utc"].strftime("%H:%M:%S.%f UTC")[:-3] +
                    f"  (Δ {p['offset_ms']:+.1f} ms)")
        rtt_s = f"{p['rtt_ms']:.0f} ms" if p["rtt_ms"] >= 0 else "offline"
        tbl.add_row("NTP/RTT", f"{rtt_s} · sync {p['sync']} · mono {p['mono']:.0f}s")
        title = "⏰ time MAX [ns + NTP]"
    else:
        tbl.add_row("Local", _fmt12(p["local"]))
        tbl.add_row("UTC", p["utc"].strftime("%H:%M:%S UTC"))
        rtt_s = f"{p['rtt_ms']:.0f}ms" if p["rtt_ms"] >= 0 else "off"
        tbl.add_row("NTP", f"Δ {p['offset_ms']:+.0f}ms · {rtt_s} · {p['sync']}")
        title = "⏰ time [12h · ms]"
    return Panel(tbl, title=title, border_style="magenta", box=box.ROUNDED)


def _fmt12(dt: _dt.datetime) -> str:
    """12h clock with 3-decimal ms: 04:25:15.298 PM. No slice bugs."""
    return dt.strftime("%I:%M:%S") + "." + dt.strftime("%f")[:3] + dt.strftime(" %p")


def time_strip() -> Text:
    """Single-line clock for dense layouts. 12h + ms."""
    p = precise_time()
    t = Text()
    t.append(_fmt12(p["local"]), style="bold cyan")
    t.append(f"  Δ{p['offset_ms']:+.0f}ms", style="magenta")
    return t


# ------------------------------------------------------- market data
# Path: OpenBB yfinance (free, no key) -> yfinance direct -> Stooq CSV.
# TTL caches keep us under Yahoo rate limits and make redraws instant.

Quote = dict  # lightweight; avoids dataclass overhead in hot loop

_QCACHE: dict = {}
_QTTL = 60.0
_HCACHE: dict = {}
_HTTL = 900.0
_NCACHE: dict = {}
_NTTL = 600.0


def _cache_get(d: dict[str, Any], k: str, ttl: float) -> Any | None:
    """Get a cached value when fresh, else None."""
    v = d.get(k)
    if v and (time.monotonic() - v[0] < ttl):
        return v[1]
    return None


def _cache_put(d: dict[str, Any], k: str, v: Any, limit: int = 128) -> None:
    """Put a value in a TTL cache, evicting oldest when full."""
    if len(d) >= limit:
        d.pop(next(iter(d)))
    d[k] = (time.monotonic(), v)


def _obb() -> Any:
    """OpenBB app, typed Any.

    Its factory returns BaseApp|Extensions union
    that mypy can't narrow — all providers used here are free/no-key.
    """
    from openbb import obb  # lazy: import costs ~1-2s, skip unless needed
    return obb


def _to_float(x: Any) -> float | None:
    """Convert to float, returning None for missing or NaN inputs."""
    try:
        if x is None:
            return None
        f = float(x)
    except Exception:
        return None
    else:
        return None if math.isnan(f) else f


def quote_via_openbb(symbol: str) -> dict | None:
    """Fetch a quote via OpenBB yfinance provider, free and keyless."""
    try:
        r = _obb().equity.price.quote(symbol, provider="yfinance")
        rows = r.results
        if not rows:
            return None
        q = rows[0]
        last = _to_float(getattr(q, "last_price", None))
        prev = _to_float(getattr(q, "prev_close", None))
        if last is None:  # some symbols expose close instead
            last = _to_float(getattr(q, "close", None))
        chg = None
        if last is not None and prev:
            chg = (last - prev) / prev * 100.0
        # OpenBB yfinance quote has NO market_cap field — merge from
        # yfinance fast_info (free, no key, ~ms, cached by Yahoo).
        mcap = None
        try:
            import yfinance as yf
            mcap = _to_float(getattr(yf.Ticker(symbol).fast_info,
                                     "market_cap", None))
        except Exception:
            mcap = None
        return {
            "symbol": symbol,
            "name": getattr(q, "name", None) or symbol,
            "price": last, "prev": prev, "chg": chg,
            "open": _to_float(getattr(q, "open", None)),
            "high": _to_float(getattr(q, "high", None)),
            "low": _to_float(getattr(q, "low", None)),
            "vol": getattr(q, "volume", None),
            "yhigh": _to_float(getattr(q, "year_high", None)),
            "ylow": _to_float(getattr(q, "year_low", None)),
            "avgvol": getattr(q, "volume_average", None),
            "mcap": mcap,
            "currency": getattr(q, "currency", None) or "",
            "src": "openbb:yfinance",
        }
    except Exception:
        return None


def quote_via_yfinance(symbol: str) -> dict | None:
    """Fetch a quote directly via yfinance, free and keyless."""
    try:
        import yfinance as yf  # lazy
        t = yf.Ticker(symbol)
        fi = t.fast_info
        last = _to_float(getattr(fi, "last_price", None))
        prev = _to_float(getattr(fi, "previous_close", None))
        if last is None:
            h = t.history(period="2d", auto_adjust=False)
            if len(h):
                last = float(h["Close"].iloc[-1])
                prev = float(h["Close"].iloc[-2]) if len(h) > 1 else prev
        chg = (last - prev) / prev * 100.0 if (last and prev) else None
        try:
            info = t.get_info() if hasattr(t, "get_info") else {}
        except Exception:
            info = {}
        return {
            "symbol": symbol,
            "name": (info.get("longName") or info.get("shortName") or symbol)
            if isinstance(info, dict) else symbol,
            "price": last, "prev": prev, "chg": chg,
            "open": _to_float(getattr(fi, "open", None)),
            "high": _to_float(getattr(fi, "day_high", None)),
            "low": _to_float(getattr(fi, "day_low", None)),
            "vol": getattr(fi, "last_volume", None),
            "yhigh": _to_float(getattr(fi, "year_high", None)),
            "ylow": _to_float(getattr(fi, "year_low", None)),
            "avgvol": None,
            "mcap": _to_float(getattr(fi, "market_cap", None)),
            "currency": getattr(fi, "currency", "") or "",
            "src": "yfinance",
        }
    except Exception:
        return None


def _require_https(url: str) -> None:
    """Validate that a URL uses HTTPS before opening it."""
    if not url.startswith("https://"):
        raise ValueError(f"refusing non-https url: {url!r}")


def quote_via_stooq(symbol: str) -> dict | None:
    """Free Stooq CSV, no key. Best-effort fallback."""
    try:
        import urllib.request as _u
        m = {"^GSPC": "spx", "^NDX": "ndq", "^NSEI": "^nsebn",
             "^BSESN": "^snx", "BTC-USD": "btcusd"}
        s = m.get(symbol, symbol)
        if "." not in s and not s.startswith("^") and "-" not in s:
            s += ".us"
        s = s.replace("-", "").replace("^", "^").lower()
        url = f"https://stooq.com/q/l/?s={s}&f=sd2t2ohlcv&h&e=csv"
        _require_https(url)
        with _u.urlopen(url, timeout=8) as r:  # noqa: S310
            line = r.read().decode().strip().splitlines()
        if len(line) < 2:
            return None
        p = line[1].split(",")
        close = _to_float(p[6]) if len(p) > 6 else None
        if close is None or close == 0:
            return None
        return {"symbol": symbol, "name": symbol, "price": close,
                "prev": None, "chg": None,
                "open": _to_float(p[3]), "high": _to_float(p[4]),
                "low": _to_float(p[5]), "vol": p[7] if len(p) > 7 else None,
                "yhigh": None, "ylow": None, "avgvol": None,
                "mcap": None, "currency": "", "src": "stooq"}
    except Exception:
        return None


def _fill_missing(q: dict, src: dict, keys: tuple[str, ...]) -> None:
    """Fill missing numerics in a quote from a fallback source."""
    for k in keys:
        if q.get(k) is None and src.get(k) is not None:
            q[k] = src[k]


def _apply_yf_fallback(symbol: str, q: dict | None) -> dict | None:
    """Merge yfinance fields when OpenBB lacks a price."""
    if q is not None and q.get("price") is not None:
        return q
    yq = quote_via_yfinance(symbol)
    if yq is None:
        return q
    if q is None:
        return yq
    if yq.get("price") is not None:
        # keep OpenBB name/src but fill missing numerics from yf
        _fill_missing(
            q, yq, ("price", "prev", "chg", "open", "high", "low",
                    "vol", "mcap", "currency"),
        )
        q["src"] = "openbb+yf"
    return q


def _apply_stooq_fallback(symbol: str, q: dict | None) -> dict | None:
    """Merge Stooq fields when price is still missing."""
    if q is not None and q.get("price") is not None:
        return q
    sq = quote_via_stooq(symbol)
    if sq is None or sq.get("price") is None:
        return q
    if q is None:
        return sq
    _fill_missing(q, sq, ("price", "open", "high", "low", "vol"))
    return q


def _apply_history_fallback(symbol: str, q: dict | None) -> dict | None:
    """Use latest history close as a last resort."""
    if q is not None and q.get("price") is not None:
        return q
    try:
        h = fetch_history(symbol, "5d")
    except Exception:
        return q
    else:
        if h is None or not len(h):
            return q
        px = float(h[-1])
        if not q:
            return {"symbol": symbol, "name": symbol, "price": px,
                    "prev": float(h[-2]) if len(h) > 1 else None,
                    "chg": None, "open": None, "high": None,
                    "low": None, "vol": None, "yhigh": None,
                    "ylow": None, "avgvol": None, "mcap": None,
                    "currency": "", "src": "history"}
        q["price"] = px
        if q.get("prev"):
            q["chg"] = (px - q["prev"]) / q["prev"] * 100.0
        q["src"] += "+hist"
        return q


def _offline_quote(symbol: str) -> dict:
    """Build an offline placeholder quote."""
    return {"symbol": symbol, "name": symbol, "price": None, "prev": None,
            "chg": None, "open": None, "high": None, "low": None,
            "vol": None, "yhigh": None, "ylow": None, "avgvol": None,
            "mcap": None, "currency": "", "src": "offline"}


def fetch_quote(symbol: str) -> dict:
    """Fetch a quote with OpenBB, Yahoo, Stooq, then history fallbacks."""
    c = _cache_get(_QCACHE, symbol, _QTTL)
    if c:
        return c
    # OpenBB dict may have price=None for indices (^GSPC) — fall through
    # and merge best fields. Order: openbb -> yfinance -> stooq.
    q = quote_via_openbb(symbol)
    q = _apply_yf_fallback(symbol, q)
    q = _apply_stooq_fallback(symbol, q)
    q = _apply_history_fallback(symbol, q)
    if not q:
        q = _offline_quote(symbol)
    _cache_put(_QCACHE, symbol, q)
    return q


def fetch_batch(symbols: list[str]) -> list[dict]:
    """12 workers == i5-1335U threads. Order-preserving."""
    syms = list(dict.fromkeys(symbols))
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        return list(ex.map(fetch_quote, syms))


_FCACHE: dict = {}
_FTTL = 3600.0


def _sane_yield(x: Any) -> float | None:
    """Yahoo sometimes mixes ratio vs percent for yields.

    Cap absurdity:
    0.46 for NVDA means 0.46%, never 46%.
    """
    v = _to_float(x)
    if v is None:
        return None
    return v / 100.0 if v > 0.25 else v


def _fmp_get(endpoint: str, symbol: str, params: str = "") -> object | None:
    """FMP stable REST, only when user gave a key (FMP_API_KEY env).

    Free Basic = 250 calls/day. Never imported/used without a key.
    """
    import json as _j
    import urllib.request as _u
    key = os.environ.get("FMP_API_KEY", "").strip()
    if not key:
        return None
    try:
        url = (f"https://financialmodelingprep.com/stable/{endpoint}"
               f"?symbol={symbol}{params}&apikey={key}")
        _require_https(url)
        req = _u.Request(url, headers={"User-Agent": "VaibhavDojo/1.0"})  # noqa: S310
        with _u.urlopen(req, timeout=10) as r:  # noqa: S310
            return _j.loads(r.read().decode())
    except Exception:
        return None


def _fmp_pick(p0: dict, *ks: str) -> object | None:
    """Pick the first present key from an FMP row."""
    return next((p0.get(k) for k in ks if p0.get(k) is not None), None)


def _fmp_apply_profile(out: dict) -> bool:
    """Apply FMP profile fields, returning True when present."""
    prof = _fmp_get("profile", out["symbol"])
    if not (isinstance(prof, list) and prof):
        return False
    p0 = prof[0]
    out.update(
        name=_fmp_pick(p0, "companyName"),
        mcap=_to_float(_fmp_pick(p0, "mktCap")),
        beta=_to_float(_fmp_pick(p0, "beta")),
        y52h=_to_float(
            _fmp_pick(p0, "fiftyTwoWeekHigh") or _fmp_pick(p0, "yearHigh"),
        ),
        y52l=_to_float(
            _fmp_pick(p0, "fiftyTwoWeekLow") or _fmp_pick(p0, "yearLow"),
        ),
        div=_sane_yield(_fmp_pick(p0, "dividendYield")),
    )
    return True


def _fmp_apply_ratios(out: dict) -> None:
    """Apply FMP ratio fields to the output dict."""
    rat = _fmp_get("ratios", out["symbol"], "&limit=1")
    if not (isinstance(rat, list) and rat):
        return
    r0 = rat[0]
    out.update(
        pe=_to_float(r0.get("priceEarningsRatio")),
        fpe=_to_float(r0.get("forwardPriceEarningsRatio")),
        peg=_to_float(r0.get("pegRatio")),
        pb=_to_float(r0.get("priceToBookRatio")),
        roe=_to_float(r0.get("returnOnEquity")),
        roa=_to_float(r0.get("returnOnAssets")),
        de=_to_float(r0.get("debtEquityRatio")),
        current=_to_float(r0.get("currentRatio")),
        grossm=_to_float(r0.get("grossProfitMargin")),
        opm=_to_float(r0.get("operatingProfitMargin")),
        margins=_to_float(r0.get("netProfitMargin")),
        payout=_to_float(r0.get("payoutRatio")),
    )


def _fmp_apply_statements(out: dict) -> None:
    """Apply FMP quarterly income and balance fields."""
    inc = _fmp_get("income-statement", out["symbol"], "&period=quarter&limit=1")
    if isinstance(inc, list) and inc:
        i0 = inc[0]
        out["rev_qtr"] = _to_float(i0.get("revenue"))
        out["net_qtr"] = _to_float(i0.get("netIncome"))
        out["qtr_date"] = str(i0.get("date", ""))
        out["eps"] = _to_float(i0.get("eps"))
    bs = _fmp_get(
        "balance-sheet-statement", out["symbol"], "&period=quarter&limit=1",
    )
    if isinstance(bs, list) and bs:
        out["cash"] = _to_float(bs[0].get("cashAndCashEquivalents"))
        out["debt"] = _to_float(bs[0].get("totalDebt"))


def _fundamentals_from_fmp(_symbol: str, out: dict) -> bool:
    """Try FMP path, returning True when it produced usable data."""
    try:
        if not _fmp_apply_profile(out):
            return False
        _fmp_apply_ratios(out)
        _fmp_apply_statements(out)
    except Exception:
        return False
    else:
        out["src"] = "fmp"
        out["ok"] = bool(
            out.get("rev_qtr") or out.get("pe") or out.get("mcap"),
        )
        return bool(out["ok"])


def _yahoo_apply_info(out: dict, info: dict) -> None:
    """Apply yfinance info dict to the output."""
    out.update(
        mcap=_to_float(info.get("marketCap")),
        eps=_to_float(info.get("trailingEps")),
        pe=_to_float(info.get("trailingPE")),
        fpe=_to_float(info.get("forwardEps") and info.get("forwardPE")),
        peg=_to_float(info.get("pegRatio")),
        pb=_to_float(info.get("priceToBook")),
        roe=_to_float(info.get("returnOnEquity")),
        roa=_to_float(info.get("returnOnAssets")),
        de=_to_float(info.get("debtToEquity")),
        current=_to_float(info.get("currentRatio")),
        grossm=_to_float(info.get("grossMargins")),
        opm=_to_float(info.get("operatingMargins")),
        margins=_to_float(info.get("profitMargins")),
        div=_sane_yield(info.get("dividendYield")),
        payout=_to_float(info.get("payoutRatio")),
        beta=_to_float(info.get("beta")),
        y52h=_to_float(info.get("fiftyTwoWeekHigh")),
        y52l=_to_float(info.get("fiftyTwoWeekLow")),
        target=_to_float(info.get("targetMeanPrice")),
        shares=_to_float(info.get("sharesOutstanding")),
        rev_ttm=_to_float(info.get("totalRevenue")),
        name=info.get("longName") or info.get("shortName"),
    )


def _yahoo_apply_income(out: dict, ticker: Any) -> None:
    """Apply quarterly income rows to the output."""
    try:
        qf = ticker.quarterly_financials
    except Exception:
        return
    try:
        if qf is None or not len(qf.columns):
            return
        col = qf.columns[0]
        out["qtr_date"] = (
            str(col.date()) if hasattr(col, "date") else str(col)
        )
        for row in ("Total Revenue", "Net Income",
                    "Net Income Common Stockholders"):
            if row in qf.index:
                v = _to_float(qf.loc[row, col])
                if v is not None:
                    if "Revenue" in row:
                        out["rev_qtr"] = v
                    else:
                        out.setdefault("net_qtr", v)
    except Exception:
        pass


def _yahoo_apply_balance(out: dict, ticker: Any) -> None:
    """Apply quarterly balance rows to the output."""
    try:
        bs = ticker.quarterly_balance_sheet
    except Exception:
        return
    try:
        if bs is None or not len(bs.columns):
            return
        col = bs.columns[0]
        for row in (
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
        ):
            if row in bs.index:
                v = _to_float(bs.loc[row, col])
                if v is not None:
                    out["cash"] = v
                    break
        for row in ("Total Debt",):
            if row in bs.index:
                out["debt"] = _to_float(bs.loc[row, col])
    except Exception:
        pass


def _fundamentals_from_yahoo(symbol: str, out: dict) -> None:
    """Fill output via yfinance info plus quarterly statements."""
    try:
        import yfinance as yf
    except Exception:
        return
    try:
        ticker = yf.Ticker(symbol)
        try:
            info = (
                ticker.get_info()
                if hasattr(ticker, "get_info")
                else (ticker.info or {})
            )
        except Exception:
            info = {}
        if isinstance(info, dict):
            _yahoo_apply_info(out, info)
        _yahoo_apply_income(out, ticker)
        _yahoo_apply_balance(out, ticker)
    except Exception:
        pass
    else:
        out["src"] = "yahoo"
        out["ok"] = bool(
            out.get("rev_ttm")
            or out.get("rev_qtr")
            or out.get("eps")
            or out.get("pe"),
        )


def fetch_fundamentals(symbol: str) -> dict:
    """Fetch free fundamentals with FMP then Yahoo fallbacks.

    Fields: mcap/eps/pe/fpe/peg/pb/roe/roa/de/current/grossm/opm/margins/
    div/payout/beta/y52h/y52l/target/shares/cash/debt/rev_ttm/rev_qtr/net_qtr.
    Never raises — returns {'ok': False} offline.
    """
    c = _cache_get(_FCACHE, symbol, _FTTL)
    if c is not None:
        return c
    out: dict = {"symbol": symbol.upper(), "ok": False, "src": "yahoo"}
    # --- path 1: FMP (only with user's own free key, 250/day) ---
    if _fundamentals_from_fmp(symbol, out):
        _cache_put(_FCACHE, symbol, out)
        return out
    # --- path 2: yfinance info + quarterly income (free, no key) ---
    _fundamentals_from_yahoo(symbol, out)
    _cache_put(_FCACHE, symbol, out)
    return out


def fetch_history(symbol: str, period: str = "3mo") -> np.ndarray | None:
    """Fetch closing-price history for a symbol and period."""
    o = fetch_ohlc(symbol, period)
    return o["close"] if o else None


_OCACHE: dict = {}
_OTTL = 900.0
_ICACHE: dict = {}
_ITTL = 60.0  # intraday: fresh minute bars every data cycle -> real-time


def fetch_ohlc(symbol: str, period: str = "6mo") -> dict | None:
    """Own OHLC fetcher (yfinance, free). Cached. NumPy AVX2 arrays."""
    key = "ohlc|" + symbol + "|" + period
    c = _cache_get(_OCACHE, key, _OTTL)
    if c is not None:
        return c
    try:
        import yfinance as yf
        h = yf.Ticker(symbol).history(period=period, auto_adjust=False)
        if h is None or len(h) < 2:
            return None
        o = {
            "open": np.asarray(h["Open"], dtype=np.float64),
            "high": np.asarray(h["High"], dtype=np.float64),
            "low": np.asarray(h["Low"], dtype=np.float64),
            "close": np.asarray(h["Close"], dtype=np.float64),
            "vol": np.asarray(h["Volume"], dtype=np.float64),
        }
        m = np.isfinite(o["close"]) & np.isfinite(o["high"]) & np.isfinite(o["low"])
        if int(m.sum()) < 2:
            return None
        o = {k: v[m] for k, v in o.items()}
    except Exception:
        return None
    else:
        _cache_put(_OCACHE, key, o)
        return o


def fetch_intra(symbol: str) -> dict | None:
    """Real-time 1-minute bars, today only (yfinance free, 7d window).

    Cached 60s so every live-loop cycle can pull fresh bars while the
    market moves. After hours returns the day's bars + closed=True.
    """
    key = "intra|" + symbol
    c = _cache_get(_ICACHE, key, _ITTL)
    if c is not None:
        return c
    try:
        import yfinance as yf
        h = yf.Ticker(symbol).history(period="1d", interval="1m",
                                       auto_adjust=False)
        if h is None or len(h) < 2:
            return None
        o = {
            "open": np.asarray(h["Open"], dtype=np.float64),
            "high": np.asarray(h["High"], dtype=np.float64),
            "low": np.asarray(h["Low"], dtype=np.float64),
            "close": np.asarray(h["Close"], dtype=np.float64),
            "vol": np.asarray(h["Volume"], dtype=np.float64),
        }
        m = np.isfinite(o["close"])
        if int(m.sum()) < 2:
            return None
        o = {k: v[m] for k, v in o.items()}
        try:
            last_ts = h.index[m][-1]
            age_min = (pd_now(last_ts) - last_ts).total_seconds() / 60.0
        except Exception:
            age_min = 999.0
        o["age_min"] = age_min
        o["closed"] = age_min > 90
    except Exception:
        return None
    else:
        _cache_put(_ICACHE, key, o)
        return o


def pd_now(ts: Any) -> Any:
    """Now in the bar index's tz (or naive UTC). Tiny helper, no deps."""
    import pandas as pd
    try:
        tz = ts.tz
    except Exception:
        tz = None
    try:
        return pd.Timestamp.now(tz=tz)
    except Exception:
        return pd.Timestamp.utcnow()


# ------------------------------------------- AVX2 vectorized indicators
# All O(n), no Python loops — NumPy/OpenBLAS uses AVX2+FMA on this CPU.

def sma(a: np.ndarray, n: int) -> np.ndarray:
    """Compute a simple moving average over the array."""
    if len(a) < n:
        return np.full(len(a), np.nan)
    c = np.cumsum(a, dtype=np.float64)
    c[n:] = c[n:] - c[:-n]
    out = c[n - 1:] / n
    return np.concatenate((np.full(n - 1, np.nan), out))


def ema(a: np.ndarray, n: int) -> np.ndarray:
    """Compute an exponential moving average over the array."""
    k = 2.0 / (n + 1.0)
    out = np.empty_like(a)
    out[0] = a[0]
    # single pass, numpy ufunc accumulate would also work; loop here is
    # tiny (n~90) vs network ms — keep exact Wilder-style EMA.
    for i in range(1, len(a)):
        out[i] = a[i] * k + out[i - 1] * (1 - k)
    return out


def rsi(a: np.ndarray, n: int = 14) -> float | None:
    """Compute the RSI momentum oscillator for the series."""
    if len(a) < n + 1:
        return None
    d = np.diff(a)  # AVX2
    gains = np.where(d > 0, d, 0.0)
    downs = np.where(d < 0, -d, 0.0)
    ag = np.mean(gains[:n])
    al = np.mean(downs[:n])
    for i in range(n, len(gains)):  # Wilder smoothing, short loop
        ag = (ag * (n - 1) + gains[i]) / n
        al = (al * (n - 1) + downs[i]) / n
    if al == 0:
        return 100.0
    return float(100 - 100 / (1 + ag / al))


_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(a: np.ndarray | None, w: int = 24) -> str:
    """Render a compact sparkline for a price series."""
    if a is None or len(a) == 0:
        return "—"
    if len(a) > w:  # vectorized resample: mean of buckets
        idx = np.linspace(0, len(a), w + 1).astype(np.int64)
        v = np.array([a[idx[i]:idx[i + 1]].mean()
                      if idx[i + 1] > idx[i] else a[idx[i]]
                      for i in range(w)])
    else:
        v = a
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi == lo:
        return "─" * len(v)
    pos = ((v - lo) / (hi - lo) * 7).astype(np.int64)
    return "".join(_BLOCKS[int(p)] for p in pos)


def fmt_money(x: Any) -> str:
    """Format a money value with T/B/M/K suffixes."""
    if x is None:
        return "—"
    try:
        f = float(x)
    except Exception:
        return str(x)
    else:
        a = abs(f)
        for thresh, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if a >= thresh:
                return f"{f / thresh:.2f}{suffix}"
        return f"{f:,.2f}"


def fmt_px(x: Any, cur: str = "") -> str:
    """Format a price with an optional currency suffix."""
    if x is None:
        return "—"
    s = f"{float(x):,.2f}"
    return f"{s} {cur}".strip()


# ---------------------------------------------------------------- news
# 100% FREE, no keys: OpenBB company news (provider=yfinance) +
# yfinance get_news + Yahoo RSS. biztoc/world needs a key -> skipped.
def _rss_news(feed_sym: str, limit: int) -> list[dict]:
    """Yahoo Finance RSS — free, keyless, ~200ms. Best world fallback."""
    try:
        import html as _h
        import urllib.request as _u
        import xml.etree.ElementTree as ET
        url = (f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={feed_sym}"
               f"&region=US&lang=en-US")
        _require_https(url)
        req = _u.Request(url, headers={"User-Agent": "Mozilla/5.0"})  # noqa: S310
        with _u.urlopen(req, timeout=8) as r:  # noqa: S310
            root = ET.fromstring(r.read())  # noqa: S314
        out = []
        for it in root.iter("item"):
            t = it.findtext("title", "")
            if t:
                out.append({
                    "title": _h.unescape(t)[:180],
                    "pub": (it.findtext("pubDate", "") or "")[:16],
                    "src": "Yahoo RSS",
                    "link": it.findtext("link", ""),
                })
            if len(out) >= limit:
                break
    except Exception:
        return []
    else:
        return out


def _news_from_openbb(symbol: str | None, limit: int, out: list[dict]) -> None:
    """Append OpenBB company news with free yfinance provider."""
    try:
        if symbol and not symbol.startswith("^"):
            r = _obb().news.company(symbol, provider="yfinance", limit=limit)
            for n in (r.results or [])[:limit]:
                title = getattr(n, "title", None) or "?"
                out.append({
                    "title": str(title)[:180],
                    "pub": str(getattr(n, "date", ""))[:16],
                    "src": str(getattr(n, "source", "Yahoo")),
                    "link": str(getattr(n, "url", "")),
                })
    except Exception:
        pass


def _news_from_yfinance(symbol: str | None, limit: int, out: list[dict]) -> None:
    """Append yfinance direct news when still short."""
    try:
        import yfinance as yf
    except Exception:
        return
    if not symbol or symbol.startswith("^") or len(out) >= limit:
        return
    try:
        items = yf.Ticker(symbol).get_news() or []
    except Exception:
        return
    for n in items[:limit]:
        ct = n.get("content", n) if isinstance(n, dict) else {}
        if not isinstance(ct, dict):
            continue
        title = str(ct.get("title", "?"))[:180]
        if all(title != o["title"] for o in out):
            out.append({
                "title": title,
                "pub": str(ct.get("pubDate", ""))[:16],
                "src": ((ct.get("provider", {}) or {})
                        .get("displayName", "Yahoo")),
                "link": ((ct.get("clickThroughUrl", {}) or {})
                         .get("url", "")),
            })


def _news_from_rss(symbol: str | None, limit: int, out: list[dict]) -> None:
    """Append Yahoo RSS headlines when still short."""
    if len(out) >= limit:
        return
    feeds = [symbol or "^GSPC", "AAPL", "^GSPC"]
    for feed in feeds:
        if len(out) >= limit:
            break
        for item in _rss_news(feed, limit):
            if all(item["title"] != o["title"] for o in out):
                out.append(item)
            if len(out) >= limit:
                break


def fetch_news(symbol: str | None = None, limit: int = 6) -> list[dict]:
    """Fetch news via OpenBB, yfinance, then Yahoo RSS fallbacks."""
    key = (symbol or "WORLD") + f"|{limit}"
    c = _cache_get(_NCACHE, key, _NTTL)
    if c is not None:
        return c
    out: list[dict] = []
    # 1) OpenBB company news with FREE yfinance provider (verified no-key)
    _news_from_openbb(symbol, limit, out)
    # 2) yfinance get_news direct (free)
    _news_from_yfinance(symbol, limit, out)
    # 3) Yahoo RSS fallback — company feed, else market headline feed
    _news_from_rss(symbol, limit, out)
    trimmed = out[:limit]
    _cache_put(_NCACHE, key, trimmed)
    return trimmed


# -------------------------------------------------------------- render

def _chg_cell(chg: float | None) -> tuple[str, str]:
    """Format a change cell with arrow and style."""
    if chg is None:
        return "—", "dim"
    arrow = "▲" if chg >= 0 else "▼"
    style = "bold green" if chg >= 0 else "bold red"
    return f"{arrow}{chg:+.1f}%", style


def _sanitize_hl(q: dict) -> tuple[Any, Any]:
    """Hide stale H/L rather than show high<price."""
    hi, lo, pxv = q.get("high"), q.get("low"), q.get("price")
    if hi is not None and pxv is not None:
        try:
            if float(hi) < float(pxv) * 0.99 or (
                    lo is not None and float(lo) > float(pxv) * 1.01):
                return None, None
        except Exception:
            pass
    return hi, lo


def _format_sym(symbol: str, idx: int, sel: int) -> tuple[str, str | None]:
    """Format a symbol with selection marker and style."""
    sym = symbol.replace(".NS", "").lstrip("^")
    if len(sym) > 10:
        sym = sym[:9] + "·"
    if idx == sel:
        return f"▶{sym}", "on rgb(40,30,70)"
    return f" {sym}", None


def _quote_row(
    q: dict,
    histories: dict,
    tw: int,
    idx: int,
    sel: int,
    *,
    show_hl: bool,
    show_extra: bool,
) -> tuple[list[Any], str | None]:
    """Build one quote-table row with style."""
    chg_s, st = _chg_cell(q.get("chg"))
    px = _short_px(q.get("price"))
    sp = sparkline(histories.get(q["symbol"]), w=tw)
    hi, lo = _sanitize_hl(q)
    hl = f"{_short_px(hi)}/{_short_px(lo)}"
    w52 = f"{fmt_money(q.get('yhigh'))}/{fmt_money(q.get('ylow'))}"
    vols = _short_vol(q.get("vol"))
    sym, row_style = _format_sym(q["symbol"], idx, sel)
    row: list[Any] = [sym, px, Text(chg_s, style=st), sp]
    if show_hl:
        row.append(hl)
    if show_extra:
        row += [w52, vols]
    row.append(fmt_money(q.get("mcap")))
    return row, row_style


def quotes_table(quotes: list[dict], histories: dict, sel: int = -1,
                 width: int = 80) -> Table:
    """Dense autoscale: no '#' col (▶ marks selection), short nums, no '…'.

    width here ≈ panel inner width. narrow<70: Symbol/Price/Δ/Trend only.
    """
    tbl = Table(box=box.ROUNDED, expand=True, show_lines=False,
                pad_edge=False, padding=(0, 1))
    tbl.add_column("Symbol", style="bold white", no_wrap=True, width=10)
    tbl.add_column("Price", justify="right", style="bold cyan", no_wrap=True)
    tbl.add_column("Δ%", justify="right", no_wrap=True, width=8)
    # trend eats leftover: total fixed ≈ 10+12+8+9+7+6+ paddings
    show_hl = width >= 70
    show_extra = width >= 95
    tw = max(8, min(22, width - (10 + 12 + 8 + (9 if show_hl else 0)
                                 + (13 if show_extra else 0) + 8 + 8)))
    tbl.add_column("Trend", style="magenta", no_wrap=True, width=tw)
    if show_hl:
        tbl.add_column("H/L", justify="right", style="dim", no_wrap=True, width=11)
    if show_extra:
        tbl.add_column("52W", justify="right", style="yellow", no_wrap=True, width=11)
        tbl.add_column("Vol", justify="right", style="dim", no_wrap=True, width=6)
    tbl.add_column("MCap", justify="right", style="green", no_wrap=True, width=7)
    for i, q in enumerate(quotes):
        row, row_style = _quote_row(
            q, histories, tw, i, sel, show_hl=show_hl, show_extra=show_extra,
        )
        tbl.add_row(*row, style=row_style)
    return tbl


def _short_px(x: Any) -> str:
    """Format a short price for dense quote tables."""
    if x is None:
        return "—"
    try:
        f = float(x)
    except Exception:
        return "—"
    a = abs(f)
    if a >= 10000:
        return f"{f:,.0f}"
    if a >= 1000:
        return f"{f:,.1f}"
    return f"{f:,.2f}"


def _short_vol(v: Any) -> str:
    """Format a short volume for dense quote tables."""
    try:
        f = float(v)
    except Exception:
        return "—"
    if f >= 1e9:
        return f"{f/1e9:.1f}B"
    if f >= 1e6:
        return f"{f/1e6:.0f}M"
    if f >= 1e3:
        return f"{f/1e3:.0f}K"
    return f"{f:.0f}"


def news_panel(items: list[dict], title: str, limit_w: int = 0) -> Panel:
    """Render headlines as a bordered panel."""
    if not items:
        return Panel("No headlines right now (offline?). Try again in a minute.",
                     title=title, border_style="yellow", box=box.ROUNDED)
    t = Table(show_header=False, box=None, pad_edge=False, expand=True)
    t.add_column("n", width=3, style="dim")
    t.add_column("news", style="white")
    for i, n in enumerate(items, 1):
        maxlen = 150 if not limit_w or limit_w > 100 else 90
        head = _cut_words(n["title"], maxlen)
        src = n.get("src", "")
        pub = str(n.get("pub", ""))[:16]
        t.add_row(f"{i}.", f"[bold]{head}[/]\n[dim]{src} · {pub}[/]")
    return Panel(t, title=title, border_style="yellow", box=box.ROUNDED)


# ------------------------------------------------- history chart
# AVX2 vectorized resample + min/max. Width autoscales to terminal.

PERIODS = ["1mo", "3mo", "6mo", "1y", "2y"]
PERIOD_IDX = {"1mo": 0, "3mo": 1, "6mo": 2, "1y": 3, "2y": 4}


def render_history_chart(symbol: str, period: str = "6mo",
                         width: int = 80, height: int = 12,
                         live_price: float | None = None) -> Panel:
    """Own-code OHLC candles + SMA + volume. NumPy buckets, no ext libs."""
    o = fetch_ohlc(symbol, period)
    if o is None or len(o["close"]) < 5:
        return Panel(f"No history for {symbol} ({period}).",
                     title=f"📈 {symbol} {period}", border_style="cyan",
                     box=box.ROUNDED)
    oo, hh, ll, cc, vv = o["open"], o["high"], o["low"], o["close"], o["vol"]
    if live_price:  # live extension: today's forming candle follows the quote
        lp = float(live_price)
        oo = np.append(oo, cc[-1])
        hh = np.append(hh, max(hh[-1], lp))
        ll = np.append(ll, min(ll[-1], lp))
        cc = np.append(cc, lp)
        vv = np.append(vv, 0.0)
    n = len(cc)
    return _candles_panel(
        f"📈 {symbol} candles [{period}]  "
        "[←/→ period · I live · H chart]",
                          oo, hh, ll, cc, vv, n, f"{n} sessions", width, height)


def render_intra_chart(symbol: str, width: int = 80, height: int = 12,
                       live_price: float | None = None) -> Panel:
    """Real-time 1-minute candles for today.

    Refreshed every data cycle
    (60s cache) so the right edge moves while the market is open.
    """
    o = fetch_intra(symbol)
    if o is None or len(o["close"]) < 5:
        return Panel(f"No intraday bars for {symbol} (weekend? try H daily).",
                     title=f"📈 {symbol} LIVE", border_style="cyan",
                     box=box.ROUNDED)
    oo, hh, ll, cc, vv = o["open"], o["high"], o["low"], o["close"], o["vol"]
    if live_price:
        lp = float(live_price)
        oo = np.append(oo, cc[-1])
        hh = np.append(hh, max(hh[-1], lp))
        ll = np.append(ll, min(ll[-1], lp))
        cc = np.append(cc, lp)
        vv = np.append(vv, 0.0)
    age = o.get("age_min", 999.0)
    tag = "CLOSED" if o.get("closed") else f"LIVE · bar {age:.0f}m old"
    return _candles_panel(f"📈 {symbol} 1m [{tag}]  [I daily · H chart]",
                          oo, hh, ll, cc, vv, len(cc),
                          f"{len(cc)} 1m bars", width, height)


def _candles_panel(title: str, oo: np.ndarray, hh: np.ndarray, ll: np.ndarray,
                   cc: np.ndarray, vv: np.ndarray, n: int, n_label: str,
                   width: int, height: int) -> Panel:
    """Shared candle renderer.

    Buckets OHLC into w columns (O=first, H=max,
    L=min, C=last, V=sum). Pure NumPy, no external chart lib.
    """
    w = max(20, min(width - 14, 120))
    h = max(6, min(height, 16))
    n = len(cc)
    if n > w:
        idx = np.linspace(0, n, w + 1).astype(np.int64)
        bo = np.array([oo[idx[i]] for i in range(w)])
        bh = np.array([hh[idx[i]:idx[i + 1]].max() for i in range(w)])
        bl = np.array([ll[idx[i]:idx[i + 1]].min() for i in range(w)])
        bc = np.array([cc[idx[i + 1] - 1] for i in range(w)])
        bv = np.array([vv[idx[i]:idx[i + 1]].sum() for i in range(w)])
    else:
        bo, bh, bl, bc, bv = oo, hh, ll, cc, vv
        w = n
    lo, hi = float(np.min(bl)), float(np.max(bh))
    span = (hi - lo) or 1.0
    vmax = float(np.max(bv)) or 1.0
    s = sma(bc, min(20, len(bc) // 2 or 1))
    txt = Text()
    txt.append(f"{hi:,.2f} hi  green=up red=down ·=SMA\n", style="dim")
    for r in range(h):
        for c in range(w):
            y_top = hi - (r / max(1, h - 1)) * span
            y_bot = hi - ((r + 1) / max(1, h - 1)) * span
            ch, st = " ", None
            # wick range
            if bh[c] >= y_bot and bl[c] <= y_top:
                ch, st = "│", "dim"
            # body range
            btop, bbot = max(bo[c], bc[c]), min(bo[c], bc[c])
            if btop >= y_bot and bbot <= y_top:
                up = bc[c] >= bo[c]
                ch, st = ("█", "green" if up else "red")
            # SMA dot
            if ch == " " and s[c] == s[c] and s[c] >= y_bot and s[c] <= y_top:
                ch, st = "·", "yellow"
            txt.append(ch, style=st or "")
        txt.append("\n")
    # volume footer, vectorized 0-7 blocks
    vpos = ((bv / vmax) * 7).astype(np.int64)
    txt.append("".join(_BLOCKS[int(p)] for p in vpos) + "\n", style="dim cyan")
    chg = (bc[-1] - bc[0]) / bc[0] * 100 if bc[0] else 0
    col = "green" if chg >= 0 else "red"
    txt.append(f"{lo:,.2f} lo  ", style="dim")
    txt.append(f"last {bc[-1]:,.2f}  {chg:+.1f}%  {n_label}",
               style=f"bold {col}")
    return Panel(txt, title=title, border_style="cyan", box=box.ROUNDED)


# ------------------------------------------------- keyboard
# Non-blocking arrows via termios+select. No new deps (stdlib only).

_PARSE_TABLE: dict[bytes, str] = {
    b"\x1b": "esc",
    b"\r": "enter",
    b"\n": "enter",
    b"\x7f": "back",
    b"\x08": "back",
    b"\x03": "quit",
    b"\x1b[A": "up",
    b"\x1bOA": "up",
    b"\x1b[B": "down",
    b"\x1bOB": "down",
    b"\x1b[C": "right",
    b"\x1bOC": "right",
    b"\x1b[D": "left",
    b"\x1bOD": "left",
    b"\x1b[5~": "pgup",
    b"\x1b[5;5~": "pgup",
    b"\x1b[6~": "pgdn",
    b"\x1b[6;5~": "pgdn",
    b"\x1b[H": "home",
    b"\x1b[1~": "home",
    b"\x1bOH": "home",
    b"\x1b[F": "end",
    b"\x1b[4~": "end",
    b"\x1bOF": "end",
}


class KeyReader:
    """Non-blocking keyboard on the raw fd (os.read, not buffered stdin).

    Why raw: select() on buffered sys.stdin misses bytes already in
    Python's readahead, which made arrows look dead. Handles both
    normal (ESC[A) and application-cursor (ESCOA) sequences, PgUp/PgDn.
    """

    def __init__(self) -> None:
        """Initialize with the current stdin file descriptor."""
        self.fd = sys.stdin.fileno()
        self.old: Any = None
        self._on = False

    def on(self) -> None:
        """Enable raw cbreak mode when attached to a TTY."""
        if self._on or not sys.stdin.isatty():
            return
        try:
            self.old = _termios.tcgetattr(self.fd)
            _tty.setcbreak(self.fd)
            self._on = True
        except Exception:
            self.old = None

    def off(self) -> None:
        """Restore cooked terminal mode when active."""
        if self._on and self.old is not None:
            try:
                _termios.tcsetattr(self.fd, _termios.TCSADRAIN, self.old)
            except Exception:
                pass
        self._on = False

    def __enter__(self) -> Self:
        """Enter raw-mode context."""
        self.on()
        return self

    def __exit__(self, *a: object) -> None:
        """Exit raw-mode context."""
        self.off()

    def suspend(self) -> None:
        """Restore cooked mode for input()/prompts. Use as: kr.suspend() ... kr.on()."""
        self.off()

    @staticmethod
    def parse(data: bytes) -> str | None:
        """Pure parser (unit-testable without a TTY). Returns key name."""
        if not data:
            return None
        hit = _PARSE_TABLE.get(data)
        if hit is not None:
            return hit
        try:
            ch = data[:1].decode()
        except Exception:
            return None
        else:
            return "enter" if ch in ("\r", "\n") else ch

    def get(self, timeout: float = 0.05) -> str | None:
        """Return a key name or None. Arrows are up/down/left/right."""
        import os as _os
        try:
            r, _, _ = _select.select([self.fd], [], [], timeout)
            if not r:
                return None
            data = _os.read(self.fd, 32)
            return self.parse(data)
        except Exception:
            return None


KEYS_HELP = (
    "↑↓ move · Enter dive · H chart · I live · F funds · N news · Tab view · "
    "W list · T time · / ask · D def · E explain · "
    "4=bloomberg · ? help · Q quit"
)


def _cut_words(s: str, n: int) -> str:
    """Truncate at a word boundary — never 'Same Th' mid-word cuts."""
    s = (s or "").strip()
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)
    return (cut[0] if len(cut) > 1 and len(cut[0]) > n * 0.6 else s[:n]).rstrip() + "…"


def _deep_header(symbol: str, q: dict) -> None:
    """Print the deep-dive quote header."""
    chg = q.get("chg")
    chg_s = f"{chg:+.2f}%" if chg is not None else "—"
    c = "green" if (chg or 0) >= 0 else "red"
    console.print(Panel(
        f"[bold cyan]{fmt_px(q.get('price'), q.get('currency',''))}[/]  "
        f"[bold {c}]{chg_s}[/]   O {fmt_px(q.get('open'))}  "
        f"H {fmt_px(q.get('high'))}  L {fmt_px(q.get('low'))}  "
        f"Vol {q.get('vol') or '—'}  MCap {fmt_money(q.get('mcap'))}\n"
        f"[dim]src={q.get('src')} · {q.get('name','')}[/]",
        title=f"◆ {symbol}", border_style=c, box=box.ROUNDED))


def _deep_technicals(hist: Any) -> None:
    """Print technical indicators or a fallback notice."""
    if hist is None or len(hist) <= 20:
        console.print(
            Panel(
                "Not enough history for technicals.",
                border_style="cyan",
                box=box.ROUNDED,
            ),
        )
        return
    s20, s50 = sma(hist, 20), sma(hist, 50)
    e12, e26 = ema(hist, 12), ema(hist, 26)
    macd = float(e12[-1] - e26[-1])
    rsi_v = rsi(hist)
    tbl = Table(box=box.ROUNDED, expand=True)
    tbl.add_column("Signal", style="bold white")
    tbl.add_column("Value", justify="right", style="cyan")
    tbl.add_column("Read", justify="right")
    last = float(hist[-1])
    tbl.add_row(
        "Last",
        f"{last:,.2f}",
        "[green]uptrend[/]" if last > (s20[-1] or 0) else "[red]weak[/]",
    )
    tbl.add_row(
        "SMA 20/50",
        f"{s20[-1]:,.2f} / {s50[-1]:,.2f}",
        "golden" if s20[-1] > s50[-1] else "death",
    )
    tbl.add_row(
        "MACD (12-26)",
        f"{macd:+.2f}",
        "[green]bull[/]" if macd > 0 else "[red]bear[/]",
    )
    tbl.add_row(
        "RSI 14",
        f"{rsi_v:.1f}" if rsi_v else "—",
        "overbought"
        if (rsi_v or 50) > 70
        else ("oversold" if (rsi_v or 50) < 30 else "neutral"),
    )
    tbl.add_row("6-mo trend", sparkline(hist, 40), f"{len(hist)} sessions")
    console.print(
        Panel(
            tbl,
            title="📈 technicals [NumPy AVX2 vectorized]",
            border_style="cyan",
            box=box.ROUNDED,
        ),
    )


def _deep_profile(symbol: str, profile: Any) -> None:
    """Print the company profile block when available."""
    if profile is None:
        return
    try:
        desc = (getattr(profile, "long_description", "") or "")[:600]
        tbl = Table(show_header=False, box=None, expand=True)
        tbl.add_column("k", style="bold cyan", width=16)
        tbl.add_column("v", style="white")
        tbl.add_row("Name", str(getattr(profile, "name", symbol)))
        tbl.add_row(
            "Sector",
            f"{getattr(profile, 'sector', '—')} / "
            f"{getattr(profile, 'industry_category', '—')}",
        )
        tbl.add_row(
            "Employees",
            f"{getattr(profile, 'employees', '—'):,}"
            if getattr(profile, "employees", None)
            else "—",
        )
        tbl.add_row(
            "Beta / Yield",
            f"{getattr(profile, 'beta', '—')} / "
            f"{getattr(profile, 'dividend_yield', '—')}",
        )
        tbl.add_row(
            "Shares out",
            f"{fmt_money(getattr(profile, 'shares_outstanding', None))}",
        )
        console.print(
            Panel(
                tbl,
                title="🏢 company profile [OpenBB free]",
                border_style="blue",
                box=box.ROUNDED,
            ),
        )
        if desc:
            console.print(
                Panel(
                    desc + ("…" if len(desc) == 600 else ""),
                    title="About",
                    border_style="dim",
                    box=box.ROUNDED,
                ),
            )
    except Exception:
        pass


def _fetch_profile(symbol: str) -> Any:
    """Fetch an OpenBB company profile, returning None offline."""
    try:
        r = _obb().equity.profile(symbol, provider="yfinance")
    except Exception:
        return None
    else:
        return r.results[0] if r.results else None


def deep_dive(symbol: str, theme: str, news_n: int = 8) -> None:
    """Show a deep dive for one symbol with quote and news."""
    symbol = symbol.strip().upper()
    console.print(gradient_art(theme))
    console.print(Align.center(big_title("DEEP DIVE  " + symbol, theme)))
    with console.status(f"[cyan]Fetching {symbol} (OpenBB→Yahoo→Stooq)…[/]"):
        q = fetch_quote(symbol)
        hist = fetch_history(symbol, "6mo")
        news = fetch_news(symbol, news_n)
        profile = _fetch_profile(symbol)
    # header quote
    _deep_header(symbol, q)
    _deep_technicals(hist)
    _deep_profile(symbol, profile)
    console.print(news_panel(news, f"📰 {symbol} news [free]",
                             limit_w=console.size.width))
    console.print(render_history_chart(symbol, "6mo",
                                       width=console.size.width, height=14))
    console.print("[dim]Author: Vaibhav · License: GPL-3.0-only (default, never MIT) · "
                  "H in --live switches 1mo/3mo/6mo/1y/2y with ←/→[/]")


def _hist3(s: str) -> np.ndarray | None:
    return fetch_history(s, "3mo")


def dashboard(watch: list[str], theme: str, refresh: int, loops: int) -> list[dict]:
    """Render the dashboard once or in a refresh loop."""
    def build(sel: int = -1, width: int = 80) -> tuple[Layout, list[dict]]:
        lay = Layout()
        lay.split_column(
            Layout(name="hero", size=9 if _HAS_FIGLET else 8),
            Layout(name="time", size=9),
            Layout(name="mkt", ratio=2),
            Layout(name="news", ratio=1),
            Layout(name="foot", size=3),
        )
        title = Align.center(big_title("VAIBHAV  MARKET  DOJO", theme))
        lay["hero"].update(Panel(title, subtitle="桜 anime · free data · AVX2",
                                 border_style="magenta", box=box.ROUNDED))
        lay["time"].update(time_panel(theme))
        with cf.ThreadPoolExecutor(max_workers=12) as ex:
            fq = ex.submit(fetch_batch, watch)
            fn = ex.submit(fetch_news, None, 10)
            quotes = fq.result()
            # histories in parallel too
            hmap = dict(zip(watch, ex.map(_hist3, watch), strict=False))
            news = fn.result()
        lay["mkt"].update(Panel(
            quotes_table(quotes, hmap, sel=sel, width=width),
            title="📊 live quotes [OpenBB:yfinance → Yahoo → Stooq · 60s cache]",
            border_style="cyan", box=box.ROUNDED))
        lay["news"].update(news_panel(
            news, "📰 market headlines [free]  ·  --deep SYMBOL for company dive",
            limit_w=width))
        p = precise_time()
        lay["foot"].update(Align.center(Text(
            f"author Vaibhav · final power Vaibhav · GPL-v3-only default (never MIT) · "
            f"{p['local'].strftime('%H:%M:%S')} local · Δ{p['offset_ms']:+.0f}ms NTP · "
            f"q=quit · number=deep-dive",
            style="dim")))
        return lay, quotes
    if loops == 1 and refresh <= 0:
        width = console.size.width
        lay, quotes = build(width=width)
        console.print(gradient_art(theme))
        console.print(Align.center(big_title("VAIBHAV  MARKET  DOJO", theme)))
        console.print(time_panel(theme))
        with cf.ThreadPoolExecutor(max_workers=12) as ex:
            hmap = dict(zip(watch, ex.map(_hist3, watch), strict=False))
        quotes = fetch_batch(watch)
        console.print(Panel(quotes_table(quotes, hmap, width=width),
                            title="📊 live quotes [OpenBB:yfinance → Yahoo → Stooq]",
                            border_style="cyan", box=box.ROUNDED))
        console.print(news_panel(fetch_news(None, 5), "📰 headlines [free]",
                                 limit_w=width))
        return quotes
    n = 0
    with Live(console=console, refresh_per_second=1, screen=False) as live:
        while True:
            lay, quotes = build(width=console.size.width)
            live.update(lay)
            n += 1
            if loops > 0 and n >= loops:
                break
            time.sleep(max(1, refresh))
    return quotes


# --------------------------------------- Bloomberg live loop
# Constantly-running, keyboard-driven, autoscaling. No extra deps.

def _prompt_line(prompt: str) -> str:
    console.print(f"[bold magenta]{prompt}[/]", end="")
    try:
        return input().strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def ticker_tape(quotes: list[dict], width: int = 80) -> Text:
    """One-line Bloomberg-style tape. O(n), truncated to width."""
    t = Text()
    for q in quotes:
        chg = q.get("chg")
        if chg is None:
            s, st = f"{q['symbol']} —", "dim"
        else:
            s = f"{q['symbol']} {chg:+.1f}%"
            st = "green" if chg >= 0 else "red"
        t.append("  " + s + "  ◆", style=st)
    if len(t.plain) > width:
        t = Text(t.plain[:max(0, width - 1)] + "…", style="white")
    return t


def selected_header(quotes: list[dict], sel: int, hmap: dict) -> str:
    """Build the header line for the selected quote."""
    if not quotes:
        return "📊 quotes"
    q = quotes[sel % len(quotes)]
    px = fmt_px(q.get("price"), q.get("currency", ""))
    chg = q.get("chg")
    chg_s = f"{chg:+.2f}%" if chg is not None else "—"
    sp = sparkline(hmap.get(q["symbol"]), w=14)
    return (f"📊 ▶ {q['symbol']} {px} ({chg_s})  "
            f"H {fmt_px(q.get('high'))} L {fmt_px(q.get('low'))}  {sp}  "
            f"[↑/↓ move · Enter dive · H chart · F fundamentals]")


def funds_panel(symbol: str) -> Panel:
    """Dense fundamentals grid: valuation, profit, health, holders. Free data."""
    f = fetch_fundamentals(symbol)
    t = Table(box=None, pad_edge=False, expand=True, show_header=False)
    t.add_column("a", style="bold cyan", width=22)
    t.add_column("b", style="white")
    t.add_column("c", style="bold cyan", width=22)
    t.add_column("d", style="white")
    if not f.get("ok"):
        return Panel(f"{symbol}: no fundamentals (index/crypto have no earnings).",
                     title=f"🏦 {symbol}", border_style="blue", box=box.ROUNDED)
    rows = [
        (
            ("Mcap", fmt_money(f.get("mcap"))),
            ("Revenue TTM", fmt_money(f.get("rev_ttm"))),
        ),
         ((f"Rev Q ({f.get('qtr_date','')})", fmt_money(f.get("rev_qtr"))),
          ("Net Q", fmt_money(f.get("net_qtr")))),
         (("EPS", f"{f['eps']:.2f}" if f.get("eps") is not None else "—"),
          ("P/E", f"{f['pe']:.1f}" if f.get("pe") is not None else "—")),
         (("Fwd P/E", f"{f['fpe']:.1f}" if f.get("fpe") is not None else "—"),
          ("PEG", f"{f['peg']:.2f}" if f.get("peg") is not None else "—")),
         (("ROE", f"{f['roe']*100:.0f}%" if f.get("roe") is not None else "—"),
          (
              "Margin",
              f"{f['margins'] * 100:.1f}%" if f.get("margins") is not None else "—",
          )),
         (("Debt/Eq", f"{f['de']:.0f}" if f.get("de") is not None else "—"),
          ("Cash", fmt_money(f.get("cash")))),
         (("Beta", f"{f['beta']:.2f}" if f.get("beta") is not None else "—"),
          ("Div yld", f"{f['div']*100:.2f}%" if f.get("div") else "—")),
         (("52W", f"{fmt_money(f.get('y52h'))}/{fmt_money(f.get('y52l'))}"),
          ("Target", fmt_money(f.get("target"))))]
    for (a, b), (c, d) in rows:
        t.add_row(a, b, c, d)
    src = "FMP" if f.get("src") == "fmp" else "Yahoo"
    return Panel(t, title=f"🏦 {symbol} {f.get('name','')[:30]} [{src} free]",
                 border_style="blue", box=box.ROUNDED)


class _BloomState:
    """Mutable state for the Bloomberg live loop."""

    def __init__(
        self, watch: list[str], theme: str, refresh: int, list_name: str,
    ) -> None:
        """Initialize live-loop state with watchlists and defaults."""
        lists: dict[str, list[str]] = {"custom": watch}
        for _k, _v in WATCHLISTS.items():
            lists.setdefault(_k, _v)
        lists.update(load_custom_lists())
        order = ["custom"] + [k for k in lists if k != "custom"]
        li = order.index(list_name) if list_name in order else 0
        self.lists = lists
        self.order = order
        self.li = li
        self.watch = lists[order[li]]
        self.theme = theme
        self.refresh = refresh
        self.sel = 0
        self.period = "6mo"
        self.intra = False
        self.tmode = "ms"
        self.mode = "board"
        self.ai_text = ""
        self.ai_title = "🤖 Vaibhav AI"
        self.status = "live — press ? for keys"
        self.lastkey = "—"
        self.quotes: list[dict] = []
        self.hmap: dict = {}
        self.news: list[dict] = []
        self.last_fetch = 0.0
        self.themes = list(THEMES)
        self.views = [
            ("board", "Board"),
            ("news", "News"),
            ("history", "Charts"),
            ("funds", "Funds"),
            ("ai", "AI"),
        ]


def _bloom_tab_bar(st: _BloomState) -> Text:
    """Render the view tab bar for the live loop."""
    bar = Text()
    for key, label in st.views:
        if key == st.mode:
            bar.append(f"▸{label} ", style="bold black on cyan")
        else:
            bar.append(f" {label} ", style="dim")
    return bar


def _bloom_refresh_progress(st: _BloomState) -> None:
    """Fetch with a progress bar for the first live paint."""
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
        transient=True,
    ) as pg:
        task = pg.add_task("Fetching market…", total=len(st.watch) + 2)
        with cf.ThreadPoolExecutor(max_workers=12) as ex:
            fq = ex.submit(fetch_batch, st.watch)
            fn = ex.submit(fetch_news, None, 10)
            pg.advance(task)
            futs = {
                ex.submit(fetch_history, s, "3mo"): s for s in st.watch
            }
            st.hmap = {}
            for fut in cf.as_completed(futs):
                st.hmap[futs[fut]] = fut.result()
                pg.advance(task)
            st.quotes = fq.result()
            pg.advance(task)
            try:
                st.news = fn.result()
            except Exception:
                st.news = []
    st.last_fetch = time.monotonic()


def _bloom_refresh_plain(st: _BloomState) -> None:
    """Fetch quotes, histories and news without progress."""
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        fq = ex.submit(fetch_batch, st.watch)
        fn = ex.submit(fetch_news, None, 10)
        st.quotes = fq.result()
        st.hmap = dict(
            zip(st.watch, ex.map(_hist3, st.watch), strict=False),
        )
        try:
            st.news = fn.result()
        except Exception:
            st.news = []
    st.last_fetch = time.monotonic()


def _bloom_refresh(
    st: _BloomState, *, force: bool = False, show_progress: bool = False,
) -> None:
    """Refresh quotes, histories and news when due."""
    if not force and (
        time.monotonic() - st.last_fetch < max(5, st.refresh)
    ):
        return
    if show_progress and sys.stdout.isatty():
        _bloom_refresh_progress(st)
        return
    _bloom_refresh_plain(st)


def _bloom_hero(st: _BloomState, w: int, h: int) -> Panel:
    """Build the hero panel with title, tape and tabs."""
    from rich.console import Group as _Group

    grp = _Group(
        Align.center(
            Text.assemble(
                gradient_title("VAIBHAV BLOOMBERG DOJO", st.theme),
                (
                    f"  桜 {st.order[st.li]} · {w}x{h} · "
                    f"{st.period} · {st.refresh}s",
                    "dim",
                ),
            ),
        ),
        Align.center(ticker_tape(st.quotes, w)),
        Align.center(_bloom_tab_bar(st)),
    )
    return Panel(grp, border_style="magenta", box=box.ROUNDED)


def _bloom_body_history(st: _BloomState, w: int, h: int) -> Panel:
    """Build the history/intraday chart panel."""
    sym = (
        st.quotes[st.sel % len(st.quotes)]["symbol"]
        if st.quotes
        else st.watch[0]
    )
    lp = (
        st.quotes[st.sel % len(st.quotes)].get("price")
        if st.quotes
        else None
    )
    if st.intra:
        return render_intra_chart(
            sym, width=w, height=max(8, h - 14), live_price=lp,
        )
    return render_history_chart(
        sym, st.period, width=w, height=max(8, h - 14), live_price=lp,
    )


def _bloom_body_news(st: _BloomState, w: int, h: int) -> Panel:
    """Build the headlines panel for the live loop."""
    n = max(3, (h - 14) // 3)
    return news_panel(
        (st.news or fetch_news(None, 10))[:n],
        f"📰 headlines ×{n} [free · W lists · F funds]",
        limit_w=w,
    )


def _bloom_body_ai(st: _BloomState) -> Panel:
    """Build the AI answer panel for the live loop."""
    txt = st.ai_text
    if isinstance(txt, (tuple, list)):
        txt = "\n".join(map(str, txt))
    return Panel(
        txt or "Ask with / · Define with D · Explain with E",
        title=st.ai_title,
        border_style="green",
        box=box.ROUNDED,
    )


def _bloom_body_help() -> Panel:
    """Build the keyboard-help panel."""
    tbl = Table(show_header=False, box=None, expand=True)
    tbl.add_column("k", style="bold cyan", width=22)
    tbl.add_column("v", style="white")
    for k, v in [
        ("↑/↓ or j/k", "move selection"),
        ("Enter/GO", "deep-dive selected company"),
        ("H", "chart of selected (←/→ period · I toggles LIVE 1m)"),
        ("N / T", "headlines board / accurate-time board (T toggles ms/max)"),
        ("B", "back to quote board"),
        ("Tab", "cycle Board/News/Charts/Funds/AI"),
        ("W", "cycle watchlists (mix/usa/india/crypto/all + yours)"),
        ("1/2/3/4", "sakura / tokyo / abyss / bloomberg theme"),
        ("+ / -", "faster / slower refresh (5s..120s)"),
        ("R", "refresh now"),
        ("/", "ask AI (uses selected symbol as context)"),
        ("D", "define a word (Investopedia + Wikipedia)"),
        ("F", "fundamentals of selected company (revenue, profit, EPS)"),
        ("E", "AI explains latest headlines simply"),
        ("?", "this help"),
        ("Q / Esc", "quit"),
    ]:
        tbl.add_row(k, v)
    return Panel(
        tbl,
        title="⌨ keys — everything by keyboard",
        border_style="yellow",
        box=box.ROUNDED,
    )


def _bloom_body_board(st: _BloomState, w: int, h: int) -> Panel | Layout:
    """Build the quote-board body with optional side pane."""
    if w < 100:
        return Panel(
            quotes_table(st.quotes, st.hmap, sel=st.sel, width=w),
            title=selected_header(st.quotes, st.sel, st.hmap),
            border_style="cyan",
            box=box.ROUNDED,
        )
    body = Layout()
    body.split_row(
        Layout(name="quotes", ratio=3),
        Layout(name="side", ratio=2),
    )
    body["quotes"].update(
        Panel(
            quotes_table(st.quotes, st.hmap, sel=st.sel, width=w),
            title=selected_header(st.quotes, st.sel, st.hmap),
            border_style="cyan",
            box=box.ROUNDED,
        ),
    )
    side = Layout()
    side.split_column(
        Layout(name="tm", size=3),
        Layout(name="nw", ratio=1),
    )
    side["tm"].update(
        Panel(
            Align.center(time_strip()),
            border_style="magenta",
            box=box.ROUNDED,
        ),
    )
    nn = max(2, (h - 22) // 4)
    side["nw"].update(
        news_panel(
            (st.news[:nn] if st.news else []),
            f"📰 top ×{nn} [E=explain]",
            limit_w=w,
        ),
    )
    body["side"].update(side)
    return body


def _bloom_body(st: _BloomState, w: int, h: int, lay: Layout) -> None:
    """Update the body pane for the current mode."""
    if st.mode == "history" and st.quotes:
        lay["body"].update(_bloom_body_history(st, w, h))
    elif st.mode == "funds" and st.quotes:
        sym = st.quotes[st.sel % len(st.quotes)]["symbol"]
        lay["body"].update(funds_panel(sym))
    elif st.mode == "news":
        lay["body"].update(_bloom_body_news(st, w, h))
    elif st.mode == "ai":
        lay["body"].update(_bloom_body_ai(st))
    elif st.mode == "help":
        lay["body"].update(_bloom_body_help())
    else:
        lay["body"].update(_bloom_body_board(st, w, h))


def _bloom_foot(st: _BloomState) -> Panel:
    """Build the footer panel with keys and clock."""
    cur = precise_time()
    foot = (
        f"{KEYS_HELP}\n"
        f"[{st.order[st.li]} {st.sel + 1}/{len(st.quotes)}] "
        f"{st.quotes[st.sel % len(st.quotes)]['symbol'] if st.quotes else ''} "
        f"· key:{st.lastkey} · {_fmt12(cur['local'])} "
        f"· {st.status} · Vaibhav/GPL-v3"
    )
    return Panel(
        Align.center(Text(foot, style="dim")),
        box=box.ROUNDED,
        border_style="dim",
    )


def _bloom_draw(st: _BloomState) -> Layout:
    """Render the Bloomberg live frame."""
    w = console.size.width
    h = console.size.height
    lay = Layout()
    lay.split_column(
        Layout(name="hero", size=5),
        Layout(name="body", ratio=2),
        Layout(name="foot", size=3),
    )
    lay["hero"].update(_bloom_hero(st, w, h))
    _bloom_body(st, w, h, lay)
    lay["foot"].update(_bloom_foot(st))
    return lay


def _keys_nav(st: _BloomState, k: str) -> bool:
    """Handle arrow and page navigation keys."""
    if k in ("up", "k"):
        st.sel = (st.sel - 1) % max(1, len(st.quotes))
        st.mode = "board" if st.mode in ("help", "ai") else st.mode
        st.status = (
            f"selected {st.quotes[st.sel]['symbol']}" if st.quotes else ""
        )
        return True
    if k in ("down", "j"):
        st.sel = (st.sel + 1) % max(1, len(st.quotes))
        st.mode = "board" if st.mode in ("help", "ai") else st.mode
        st.status = (
            f"selected {st.quotes[st.sel]['symbol']}" if st.quotes else ""
        )
        return True
    if k in ("pgup", "home"):
        st.sel = 0
        st.status = (
            f"selected {st.quotes[st.sel]['symbol']}" if st.quotes else ""
        )
        return True
    if k in ("pgdn", "end"):
        st.sel = max(0, len(st.quotes) - 1)
        st.status = (
            f"selected {st.quotes[st.sel]['symbol']}" if st.quotes else ""
        )
        return True
    return False


def _keys_chart(st: _BloomState, k: str, kl: str) -> bool:
    """Handle chart and period keys."""
    if kl == "h":
        st.mode = "history"
        st.status = "chart — ←/→ period · I for LIVE 1m"
        return True
    if kl == "i":
        if st.mode != "history":
            st.mode = "history"
        st.intra = not st.intra
        st.status = (
            "LIVE 1m chart" if st.intra else f"daily chart {st.period}"
        )
        return True
    if k == "left" and st.mode == "history":
        i = PERIOD_IDX.get(st.period, 2)
        st.period = PERIODS[max(0, i - 1)]
        return True
    if k == "right" and st.mode == "history":
        i = PERIOD_IDX.get(st.period, 2)
        st.period = PERIODS[min(len(PERIODS) - 1, i + 1)]
        return True
    if kl == "b":
        st.mode = "board"
        return True
    return False


def _keys_watch(st: _BloomState, k: str, kl: str) -> bool:
    """Handle watchlist, theme and refresh keys."""
    if kl == "w":
        st.li = (st.li + 1) % len(st.order)
        st.watch = st.lists[st.order[st.li]]
        st.sel = 0
        st.last_fetch = 0.0
        _bloom_refresh(st, force=True)
        st.status = f"list: {st.order[st.li]} ({len(st.watch)})"
        return True
    if kl in ("1", "2", "3", "4"):
        idx = int(kl) - 1
        if idx < len(st.themes):
            st.theme = st.themes[idx]
        st.status = f"theme {st.theme}"
        return True
    if k == "+":
        st.refresh = max(5, st.refresh - 5)
        st.status = f"refresh {st.refresh}s"
        return True
    if k == "-":
        st.refresh = min(120, st.refresh + 5)
        st.status = f"refresh {st.refresh}s"
        return True
    if kl == "r":
        _bloom_refresh(st, force=True)
        st.status = "refreshed now"
        return True
    return False


def _keys_view(st: _BloomState, k: str, kl: str) -> bool:
    """Handle view-switching keys."""
    if kl == "n":
        st.mode = "news"
        _bloom_refresh(st, force=True)
        return True
    if kl == "f":
        st.mode = "funds"
        st.status = "fundamentals — ↑/↓ switches company"
        return True
    if k == "\t":
        keys = [kk for kk, _ in st.views]
        if st.mode in keys:
            nxt = (keys.index(st.mode) + 1) % len(keys)
            st.mode = keys[nxt]
        else:
            st.mode = "board"
        st.status = f"view: {st.mode}"
        return True
    if k == "?":
        st.mode = "help"
        return True
    return False


def _handle_enter(st: _BloomState, k: str, ask: Any) -> bool:
    """Handle the Enter deep-dive key."""
    if k != "enter":
        return False
    sym = st.quotes[st.sel]["symbol"] if st.quotes else st.watch[0]
    ov = ask(f"Deep-dive [{sym}] Enter=GO, or type symbol: ")
    if ov.strip() and ov.strip().lower() != "go":
        sym = ov.strip().upper()
    console.clear()
    deep_dive(sym, st.theme)
    ask("Enter to return to live… ")
    _bloom_refresh(st, force=True)
    return True


def _handle_time(st: _BloomState, kl: str, ask: Any) -> bool:
    """Handle the time-panel key."""
    if kl != "t":
        return False
    st.tmode = "max" if st.tmode == "ms" else "ms"
    console.clear()
    console.print(gradient_art(st.theme))
    console.print(time_panel(st.theme, st.tmode))
    ask("Enter to return… ")
    st.mode = "board"
    return True


def _ai_answer(st: _BloomState, question: str) -> None:
    """Answer one AI question into live-loop state."""
    try:
        from vaibhav_ai import answer as _ans
    except Exception as e:
        st.ai_text = f"AI offline ({e}). Train: python vaibhav_ai_train.py"
        st.ai_title = "🤖 AI offline"
    else:
        try:
            sel_sym = st.quotes[st.sel]["symbol"] if st.quotes else None
            st.ai_text, st.ai_title = _ans(
                question, context_symbol=sel_sym,
            )
        except Exception as e:
            st.ai_text = f"AI offline ({e}). Train: python vaibhav_ai_train.py"
            st.ai_title = "🤖 AI offline"
    st.mode = "ai"


def _ai_define(st: _BloomState, word: str) -> None:
    """Define one word into live-loop state."""
    try:
        from vaibhav_ai import define as _def
    except Exception as e:
        st.ai_text = f"AI offline ({e})"
        st.ai_title = "📖 offline"
    else:
        try:
            st.ai_text, st.ai_title = _def(word)
        except Exception as e:
            st.ai_text = f"AI offline ({e})"
            st.ai_title = "📖 offline"
    st.mode = "ai"


def _ai_explain(st: _BloomState) -> None:
    """Explain headlines into live-loop state."""
    try:
        from vaibhav_ai import explain_news as _ex
    except Exception as e:
        st.ai_text = f"AI offline ({e}). Train: python vaibhav_ai_train.py"
        st.ai_title = "📰 offline"
    else:
        try:
            st.ai_text = _ex(st.news[:5] if st.news else fetch_news(None, 5))
            st.ai_title = "📰 headlines explained simply [5M model]"
        except Exception as e:
            st.ai_text = f"AI offline ({e}). Train: python vaibhav_ai_train.py"
            st.ai_title = "📰 offline"
    st.mode = "ai"


def _handle_ask(st: _BloomState, k: str, ask: Any) -> bool:
    """Handle the AI ask key."""
    if k != "/":
        return False
    q = ask("🤖 ask (news/market/word): ")
    if q:
        _ai_answer(st, q)
    return True


def _handle_define(st: _BloomState, kl: str, ask: Any) -> bool:
    """Handle the define-word key."""
    if kl != "d":
        return False
    word = ask("Define word: ")
    if word:
        _ai_define(st, word)
    return True


def _handle_explain(st: _BloomState, kl: str) -> bool:
    """Handle the explain-headlines key."""
    if kl != "e":
        return False
    _ai_explain(st)
    return True


def _handle_ai(st: _BloomState, k: str, kl: str, ask: Any) -> bool:
    """Handle AI ask, define and explain keys."""
    if _handle_ask(st, k, ask):
        return True
    if _handle_define(st, kl, ask):
        return True
    return _handle_explain(st, kl)


def _dispatch_key(st: _BloomState, k: str, kl: str, ask: Any) -> str | None:
    """Dispatch one key, returning 'quit' when the loop should exit."""
    if kl in ("q", "quit", "esc"):
        return "quit"
    handled = (
        _keys_nav(st, k)
        or _handle_enter(st, k, ask)
        or _keys_chart(st, k, kl)
        or _keys_view(st, k, kl)
        or _keys_watch(st, k, kl)
        or _handle_time(st, kl, ask)
        or _handle_ai(st, k, kl, ask)
    )
    return "handled" if handled else None


def bloomberg_loop(
    watch: list[str], theme: str, refresh: int = 15, list_name: str = "custom",
) -> int:
    """Full-screen looping terminal: arrows navigate, keys act. See KEYS_HELP."""
    st = _BloomState(watch, theme, refresh, list_name)
    _bloom_refresh(st, force=True, show_progress=True)
    with Live(
        console=console, screen=True, auto_refresh=False, refresh_per_second=4,
    ) as live, KeyReader() as kr:

        def _ask(prompt: str) -> str:
            # prompts need cooked mode (echo + line editing back)
            live.stop()
            kr.suspend()
            try:
                return _prompt_line(prompt)
            finally:
                kr.on()
                live.start()

        while True:
            try:
                live.update(_bloom_draw(st), refresh=True)
            except Exception as e:  # one bad frame must never kill the desk
                st.status = (
                    f"render hiccup ({type(e).__name__}) — continuing"
                )
                try:
                    live.update(
                        Panel(
                            f"[red]{e}[/]\n{st.status}",
                            title="render guard",
                            border_style="red",
                            box=box.ROUNDED,
                        ),
                        refresh=True,
                    )
                except Exception:
                    pass
            k = kr.get(timeout=0.12)
            if time.monotonic() - st.last_fetch >= max(5, st.refresh):
                _bloom_refresh(st, force=True)
            if k is None:
                continue
            st.lastkey = k
            kl = k.lower() if len(k) == 1 else k
            if _dispatch_key(st, k, kl, _ask) == "quit":
                break
    console.clear()
    return 0

def interactive_pick(quotes: list[dict], theme: str) -> None:
    """Prompt for a deep-dive symbol after a snapshot run."""
    console.print(
        "\n[bold magenta]🌸 deep-dive?[/] "
        "type number/symbol (Enter to exit): ",
        end="",
    )
    try:
        s = input().strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not s:
        return
    sym = s
    if s.isdigit():
        i = int(s) - 1
        if 0 <= i < len(quotes):
            sym = quotes[i]["symbol"]
    deep_dive(sym, theme)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all flags."""
    ap = argparse.ArgumentParser(
        description=(
            "Vaibhav Market Dojo — anime Bloomberg CLI, "
            "accurate time, free data, 5M AI."
        ),
        epilog=(
            "Author: Vaibhav (final power). "
            "License default GPL-3.0-only, never MIT. "
            "Ask Vaibhav before any public/GitHub push."
        ),
    )
    ap.add_argument(
        "--watch",
        default=",".join(DEFAULT_WATCH),
        help=(
            "comma-separated symbols, e.g. "
            "AAPL,NVDA,RELIANCE.NS,^GSPC,BTC-USD"
        ),
    )
    ap.add_argument(
        "--deep",
        default=None,
        help="deep dive into one symbol, e.g. --deep AAPL",
    )
    ap.add_argument(
        "--news",
        default=None,
        nargs="?",
        const="",
        help="news for symbol (omit value for headlines)",
    )
    ap.add_argument(
        "--history",
        default=None,
        nargs="?",
        const="AAPL",
        help="ASCII history chart, e.g. --history AAPL --period 6mo",
    )
    ap.add_argument(
        "--intra",
        default=None,
        nargs="?",
        const="AAPL",
        help="LIVE 1-minute chart, e.g. --intra NVDA",
    )
    ap.add_argument("--period", default="6mo", choices=PERIODS)
    ap.add_argument(
        "--time", action="store_true", help="show most-accurate time only",
    )
    ap.add_argument(
        "--theme",
        default="sakura",
        choices=list(THEMES),
        help="theme: sakura/tokyo/abyss/bloomberg (amber-on-black desk)",
    )
    ap.add_argument(
        "--refresh",
        type=int,
        default=0,
        help="live refresh seconds (0 = single snapshot)",
    )
    ap.add_argument(
        "--loops",
        type=int,
        default=1,
        help="live iterations (0 = infinite)",
    )
    ap.add_argument(
        "--live",
        action="store_true",
        help="Bloomberg loop: constant refresh + arrow-key control",
    )
    ap.add_argument(
        "--ask",
        default=None,
        help="ask AI a question (RAG + 5M model + web)",
    )
    ap.add_argument(
        "--chat",
        action="store_true",
        help="chat CLI with memory (follow-ups work)",
    )
    ap.add_argument(
        "--info",
        default=None,
        metavar="SYM",
        help=(
            "everything on one ticker: "
            "quote+fundamentals+technicals+news+chart"
        ),
    )
    ap.add_argument(
        "--news-n",
        type=int,
        default=8,
        help="headlines count for --news/--info",
    )
    ap.add_argument(
        "--list",
        default=None,
        help=(
            "watchlist preset: mix/usa/india/crypto/all "
            "(plus your saved ones)"
        ),
    )
    ap.add_argument(
        "--add",
        default=None,
        help="comma symbols appended to watch, e.g. --add ETH-USD,SBIN.NS",
    )
    ap.add_argument(
        "--save-list",
        default=None,
        metavar="NAME",
        help="save current watch+--add as named list (W cycles it)",
    )
    ap.add_argument(
        "--time-mode",
        default="ms",
        choices=["ms", "max"],
        help="ms: 12h clock (default) · max: ns detail",
    )
    ap.add_argument(
        "--symbol",
        default=None,
        help=(
            "context symbol for --ask, e.g. "
            "--ask 'why fall in long detail' --symbol BTC-USD"
        ),
    )
    ap.add_argument(
        "--define",
        default=None,
        help="define a word (Investopedia + Wikipedia)",
    )
    ap.add_argument(
        "--explain",
        action="store_true",
        help="AI explains latest headlines simply",
    )
    ap.add_argument(
        "--no-pick", action="store_true", help="skip deep-dive prompt",
    )
    return ap


def _build_watch(a: Any) -> list[str]:
    """Build the watchlist from --list, --watch and --add."""
    all_lists = dict(WATCHLISTS)
    all_lists.update(load_custom_lists())
    if a.list and a.list in all_lists:
        base = list(all_lists[a.list])
    else:
        base = [
            s.strip().upper() for s in a.watch.split(",") if s.strip()
        ] or DEFAULT_WATCH
    if a.add:
        extra = [s.strip().upper() for s in a.add.split(",") if s.strip()]
        base = list(dict.fromkeys([*base, *extra]))
    return base


def _maybe_save_list(a: Any, watch: list[str]) -> None:
    """Save the current watchlist when --save-list is given."""
    if not a.save_list:
        return
    save_custom_list(a.save_list, watch)
    console.print(
        f"[green]saved list '{a.save_list}' "
        f"({len(watch)} symbols). W cycles it in --live.[/]",
    )


def _run_time(a: Any) -> int | None:
    """Handle --time, returning exit code when matched."""
    if not a.time:
        return None
    console.print(gradient_art(a.theme))
    console.print(Align.center(big_title("TIME  DOJO", a.theme)))
    console.print(time_panel(a.theme, a.time_mode))
    return 0


def _run_chat(a: Any) -> int | None:
    """Handle --chat, returning exit code when matched."""
    if not a.chat:
        return None
    try:
        from vaibhav_ai import chat_loop as _chat
    except Exception as e:
        console.print(f"[yellow]chat offline ({e})[/]")
    else:
        try:
            _chat(None)
        except Exception as e:
            console.print(f"[yellow]chat offline ({e})[/]")
    return 0


def _run_info(a: Any) -> int | None:
    """Handle --info, returning exit code when matched."""
    if not a.info:
        return None
    sym = a.info.strip().upper()
    deep_dive(sym, a.theme, news_n=a.news_n)
    return 0


def _run_deep(a: Any) -> int | None:
    """Handle --deep, returning exit code when matched."""
    if not a.deep:
        return None
    deep_dive(a.deep, a.theme)
    return 0


def _run_history(a: Any) -> int | None:
    """Handle --history, returning exit code when matched."""
    if a.history is None:
        return None
    sym = (a.history or "AAPL").strip().upper()
    console.print(gradient_art(a.theme))
    console.print(
        render_history_chart(
            sym, a.period, width=console.size.width, height=14,
        ),
    )
    return 0


def _run_intra(a: Any) -> int | None:
    """Handle --intra, returning exit code when matched."""
    if a.intra is None:
        return None
    sym = (a.intra or "AAPL").strip().upper()
    console.print(gradient_art(a.theme))
    q = fetch_quote(sym)
    console.print(
        render_intra_chart(
            sym,
            width=console.size.width,
            height=16,
            live_price=q.get("price"),
        ),
    )
    return 0


def _run_news(a: Any) -> int | None:
    """Handle --news, returning exit code when matched."""
    if a.news is None:
        return None
    sym = (a.news or "").strip().upper() or None
    console.print(gradient_art(a.theme))
    console.print(
        news_panel(
            fetch_news(sym, a.news_n),
            f"📰 news {sym or 'headlines'} ×{a.news_n} [free]",
            limit_w=console.size.width,
        ),
    )
    if a.explain:
        try:
            from vaibhav_ai import explain_news as _ex
        except Exception as e:
            console.print(
                f"[yellow]AI offline ({e}). "
                "Train: python vaibhav_ai_train.py[/]",
            )
        else:
            try:
                console.print(
                    Panel(
                        _ex(fetch_news(sym, 5)),
                        title="🤖 explained simply [5M model]",
                        border_style="green",
                        box=box.ROUNDED,
                    ),
                )
            except Exception as e:
                console.print(
                    f"[yellow]AI offline ({e}). "
                    "Train: python vaibhav_ai_train.py[/]",
                )
    return 0


def _run_explain(a: Any) -> int | None:
    """Handle --explain, returning exit code when matched."""
    if not a.explain:
        return None
    try:
        from vaibhav_ai import explain_news as _ex
    except Exception as e:
        console.print(
            f"[yellow]AI offline ({e}). Train: python vaibhav_ai_train.py[/]",
        )
    else:
        try:
            console.print(
                Panel(
                    _ex(fetch_news(None, 5)),
                    title="🤖 headlines explained [5M model]",
                    border_style="green",
                    box=box.ROUNDED,
                ),
            )
        except Exception as e:
            console.print(
                f"[yellow]AI offline ({e}). "
                "Train: python vaibhav_ai_train.py[/]",
            )
    return 0


def _run_define(a: Any) -> int | None:
    """Handle --define, returning exit code when matched."""
    if not a.define:
        return None
    try:
        from vaibhav_ai import define as _def
    except Exception as e:
        console.print(
            f"[yellow]AI offline ({e}). Train: python vaibhav_ai_train.py[/]",
        )
    else:
        try:
            text, title = _def(a.define)
            console.print(
                Panel(
                    text, title=title, border_style="green", box=box.ROUNDED,
                ),
            )
        except Exception as e:
            console.print(
                f"[yellow]AI offline ({e}). "
                "Train: python vaibhav_ai_train.py[/]",
            )
    return 0


def _run_ask(a: Any) -> int | None:
    """Handle --ask, returning exit code when matched."""
    if not a.ask:
        return None
    try:
        from vaibhav_ai import answer as _ans
        from vaibhav_ai import norm_symbol as _ns
    except Exception as e:
        console.print(
            f"[yellow]AI offline ({e}). Train: python vaibhav_ai_train.py[/]",
        )
    else:
        try:
            ctx = (a.symbol or "").strip().upper() or None
            if ctx and _ns(ctx) is None:
                ctx = _ns(a.ask)
            text, title = _ans(a.ask, context_symbol=ctx or _ns(a.ask))
            console.print(
                Panel(
                    text, title=title, border_style="green", box=box.ROUNDED,
                ),
            )
        except Exception as e:
            console.print(
                f"[yellow]AI offline ({e}). "
                "Train: python vaibhav_ai_train.py[/]",
            )
    return 0


def _run_live(a: Any, watch: list[str]) -> int | None:
    """Handle --live, returning exit code when matched."""
    if not a.live:
        return None
    if not sys.stdin.isatty():
        console.print("[red]--live needs an interactive TTY.[/]")
        return 1
    return bloomberg_loop(
        watch, a.theme, refresh=a.refresh or 15, list_name=a.list or "custom",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the Vaibhav Market Dojo command-line interface."""
    ap = _build_parser()
    a = ap.parse_args(argv)
    watch = _build_watch(a)
    _maybe_save_list(a, watch)
    for handler in (
        _run_time,
        _run_chat,
        _run_info,
        _run_deep,
        _run_history,
        _run_intra,
        _run_news,
        _run_explain,
        _run_define,
        _run_ask,
    ):
        result = handler(a)
        if result is not None:
            return result
    live_result = _run_live(a, watch)
    if live_result is not None:
        return live_result
    loops = a.loops if a.refresh > 0 else 1
    quotes = dashboard(watch, a.theme, a.refresh, loops)
    if not a.no_pick and sys.stdin.isatty():
        interactive_pick(quotes, a.theme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
