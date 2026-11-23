# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 Vaibhav — final power over this project: Vaibhav.
# License default: GPL-3.0-only, NEVER MIT. Ask Vaibhav before public/GitHub.
"""Vaibhav AI: ask/define/explain.

Retrieval-first for accuracy + 5M model for simple words.

Accuracy design (honest): a 5M model alone cannot be highly accurate.
So: 1) retrieve from credible free sources (local Investopedia glossary 6.3k,
Wikipedia REST, Investopedia pages, live Yahoo news), 2) use 5M model to
rephrase simply, 3) always cite sources + learning-not-advice disclaimer.
If model file missing, template simplifier keeps answers correct.
Run: source ~/ml_env/bin/activate && python vaibhav_ai.py --define RSI
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

HERE: Path = Path(__file__).resolve().parent
DATA: Path = HERE / "data"
MODEL: Path = HERE / "models" / "vaibhav-5m.pt"
CHECK_PATH: Path = HERE / "models" / "vaibhav-check-300k.pt"
VOCAB_PATH: Path = DATA / "vocab.json"
GLOSS_PATH: Path = DATA / "glossary.json"
DISCLAIMER = "Learning only, not financial advice."
_model: Any = None
_vocab: Any = None
_CHECK: Any = None
_GLOSS: list[dict[str, Any]] | None = None


def _ensure_https(url: str) -> None:
    """Ensure URL uses HTTPS scheme."""
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS URLs are allowed.")


def _fetch_https_bytes(url: str, headers: dict[str, str]) -> bytes:
    """Fetch bytes from an HTTPS URL."""
    _ensure_https(url)
    req = urllib.request.Request(url, headers=headers)  # noqa: S310
    with urllib.request.urlopen(req, timeout=8) as resp:  # noqa: S310
        return resp.read()


def _load_glossary() -> list[dict[str, Any]]:
    """Load local glossary rows."""
    if not GLOSS_PATH.exists():
        return []
    try:
        with GLOSS_PATH.open(encoding="utf-8") as handle:
            data: Any = json.load(handle)
    except Exception:
        return []
    else:
        return data if isinstance(data, list) else []


def gloss() -> list[dict[str, Any]]:
    """Return cached glossary rows."""
    cached: Any = globals().get("_GLOSS")
    if cached is None:
        cached = _load_glossary()
        globals()["_GLOSS"] = cached
    return cached if isinstance(cached, list) else []


def _exact_term_hit(
    rows: list[dict[str, Any]], lowered: str,
) -> dict[str, Any] | None:
    """Return exact term hit if present."""
    for row in rows:
        if str(row.get("term", "")).strip().lower() == lowered:
            return row
    return None


def _phrase_hit(
    rows: list[dict[str, Any]], lowered: str,
) -> dict[str, Any] | None:
    """Return phrase-pattern hit if present."""
    for row in rows:
        term = str(row.get("term", ""))
        lowered_term = term.lower()
        if lowered in (
            f"what is {lowered_term}",
            f"define {lowered_term}",
            f"meaning of {lowered_term}",
        ):
            return row
    return None


def _tfidf_rank(
    rows: list[dict[str, Any]], query: str, count: int,
) -> list[dict[str, Any]]:
    """Rank glossary rows with TF-IDF."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [f"{r['term']} {r['text']}" for r in rows]
    vec = TfidfVectorizer(
        stop_words="english", max_features=8000,
    ).fit([*docs, query])
    matrix = vec.transform([*docs, query]).toarray()
    last = matrix[-1]
    denom = np.linalg.norm(matrix[:-1], axis=1)
    numer = matrix[:-1] @ last
    sims = numer / (denom * (np.linalg.norm(last) + 1e-9) + 1e-9)
    idx = sims.argsort()[::-1][:count]
    return [rows[i] for i in idx if sims[i] > 0.01]


def _difflib_fallback(
    rows: list[dict[str, Any]], query: str, lowered: str, count: int,
) -> list[dict[str, Any]]:
    """Return fuzzy fallback rows."""
    names = [str(r.get("term", "")) for r in rows]
    hits = difflib.get_close_matches(query, names, n=count, cutoff=0.4)
    out: list[dict[str, Any]] = []
    for hit in hits:
        for row in rows:
            if str(row.get("term", "")) == hit:
                out.append(row)
                break
    if not out:
        out = _substring_fallback(rows, lowered, count)
    return out[:count]


def _substring_fallback(
    rows: list[dict[str, Any]], lowered: str, count: int,
) -> list[dict[str, Any]]:
    """Return substring fallback rows."""
    words = [w for w in lowered.split() if len(w) > 3]
    out: list[dict[str, Any]] = []
    for row in rows:
        hay = (str(row.get("term", "")) + " " + str(row.get("text", ""))).lower()
        if any(w in hay for w in words):
            out.append(row)
        if len(out) >= count:
            break
    return out


def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Return glossary rows for query."""
    rows = gloss()
    if not rows:
        return []
    lowered = query.strip().lower()
    # 1) exact term hit (define RSI -> RSI, not Stochastic RSI)
    hit = _exact_term_hit(rows, lowered)
    if hit is not None:
        return [hit]
    hit = _phrase_hit(rows, lowered)
    if hit is not None:
        return [hit]
    try:
        ranked = _tfidf_rank(rows, query, k)
    except Exception:
        return _difflib_fallback(rows, query, lowered, k)
    else:
        return ranked


def wiki_summary(term: str) -> dict[str, str] | None:
    """Fetch Wikipedia summary for term."""
    try:
        base = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        url = base + urllib.parse.quote(term)
        raw = _fetch_https_bytes(url, {"User-Agent": "VaibhavAI/1.0"})
        item: Any = json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return None
    else:
        if not isinstance(item, dict) or not item.get("extract"):
            return None
        page: Any = item.get("content_urls", {})
        desktop: Any = page.get("desktop", {}) if isinstance(page, dict) else {}
        link = desktop.get("page", "") if isinstance(desktop, dict) else ""
        return {
            "title": str(item.get("title", term)),
            "text": str(item.get("extract", ""))[:700],
            "url": str(link),
        }


def _investopedia_paragraphs(html: bytes) -> list[str]:
    """Extract long paragraphs from HTML."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    return [
        p.get_text(" ", strip=True)
        for p in soup.select("p")
        if len(p.get_text(strip=True)) > 60
    ]


def _investopedia_for_slug(term: str, slug: str) -> dict[str, str] | None:
    """Fetch Investopedia page for slug."""
    for letter in [slug[:1], ""]:
        url: str | None = (
            f"https://www.investopedia.com/terms/{letter}/{slug}.asp"
            if letter
            else None
        )
        if not url:
            continue
        try:
            raw = _fetch_https_bytes(url, {"User-Agent": "Mozilla/5.0"})
            paras = _investopedia_paragraphs(raw)
        except Exception:
            continue
        else:
            if paras:
                return {
                    "title": term,
                    "text": " ".join(paras)[:700],
                    "url": url,
                }
    return None


def investopedia_page(term: str) -> dict[str, str] | None:
    """Fetch Investopedia snippet for term."""
    try:
        slug = re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")
        result = _investopedia_for_slug(term, slug)
    except Exception:
        return None
    else:
        return result


def _model_files_present() -> bool:
    """Check whether model files exist."""
    return MODEL.exists() and VOCAB_PATH.exists()


def _init_model_files() -> tuple[Any, tuple[Any, Any]] | None:
    """Initialize model and vocab from disk."""
    import torch

    from vaibhav_ai_train import GPTConfig, TinyGPT

    torch.set_num_threads(int(os.environ.get("VAIBHAV_THREADS", "4")))
    torch.set_num_interop_threads(1)
    with VOCAB_PATH.open(encoding="utf-8") as handle:
        vocab_data: Any = json.load(handle)
    itos: Any = vocab_data["itos"]
    stoi: Any = vocab_data["stoi"]
    ck: Any = torch.load(str(MODEL), map_location="cpu")
    cfg = GPTConfig(vocab=len(itos), block=ck.get("cfg", {}).get("block", 128))
    model: Any = TinyGPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(requires_grad=False)
    if os.environ.get("VAIBHAV_QUANT") == "1":
        try:
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8,
            )
        except Exception:
            pass
    if os.environ.get("VAIBHAV_COMPILE") == "1":
        try:
            model = torch.compile(model)
        except Exception:
            pass
    return model, (itos, stoi)


def _load_model() -> Any:
    """Load cached 5M model if available."""
    if globals().get("_model") is not None:
        return globals().get("_model")
    if os.environ.get("VAIBHAV_NO_MODEL") == "1":
        return None
    if not _model_files_present():
        return None
    try:
        loaded = _init_model_files()
    except Exception:
        return None
    else:
        if loaded is None:
            return None
        model, vocab = loaded
        globals()["_model"] = model
        globals()["_vocab"] = vocab
        return model


_COMMON: set[str] = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "what",
    "your",
    "are",
    "was",
    "were",
    "will",
    "can",
    "market",
    "price",
    "stock",
    "money",
    "bank",
    "share",
    "fund",
    "trade",
    "buy",
    "sell",
    "profit",
    "loss",
    "risk",
}


def _looks_english(txt: str) -> bool:
    """Check whether text looks like English."""
    # Gate: 500-step char model still babbles. Only accept model text that
    # looks like real words, else fall back to retrieval (accurate).
    words = re.findall(r"[A-Za-z]{2,}", txt.lower())
    if len(words) < 8:
        return False
    hit = sum(1 for x in words if x in _COMMON)
    return hit >= 3 and (hit / max(1, len(words))) >= 0.12


def simplify_with_model(prompt: str, max_new: int = 60) -> str | None:
    """Simplify prompt with 5M model."""
    model = _load_model()
    vocab: Any = globals().get("_vocab")
    if model is None or vocab is None:
        return None
    try:
        import torch

        itos: Any = vocab[0]
        stoi: Any = vocab[1]
        enc = [stoi.get(c, 0) for c in prompt[-160:]]
        idx = torch.tensor([enc], dtype=torch.long)
        with torch.inference_mode():
            out = model.generate(idx, max_new=max_new, temp=0.7, top_k=40)
            seq = out[0].tolist()
        txt = "".join(itos[i] if 0 <= i < len(itos) else "" for i in seq[len(enc):])
        txt = re.sub(r"\s+", " ", txt).strip()
        # keep only the answer-ish tail
        if "A:" in txt:
            txt = txt.split("A:")[-1].strip()
        txt = txt[:600]
        result = txt if len(txt) > 20 and _looks_english(txt) else None
    except Exception:
        return None
    else:
        return result


def _simple_words(text: str) -> str:
    """Extract first two sentences simply."""
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sents[:2])[:500]


def _fetch_define_sources(
    term: str,
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    """Fetch Wikipedia and Investopedia sources."""
    import concurrent.futures as cf

    wiki: dict[str, str] | None = None
    inv: dict[str, str] | None = None
    query = term + " (finance)" if len(term.split()) == 1 else term
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        future_wiki = ex.submit(wiki_summary, query)
        future_inv = ex.submit(investopedia_page, term)
        try:
            wiki = future_wiki.result()
        except Exception:
            wiki = None
        try:
            inv = future_inv.result()
        except Exception:
            inv = None
    return wiki, inv


def _build_define_parts(
    rows: list[dict[str, Any]],
    wiki: dict[str, str] | None,
    inv: dict[str, str] | None,
) -> tuple[list[str], list[str]]:
    """Build definition parts and sources."""
    parts: list[str] = []
    srcs: list[str] = []
    if rows:
        parts.append(f"{rows[0]['term']}: {_simple_words(rows[0]['text'])}")
        srcs.append("Investopedia glossary (local 6.3k)")
    if wiki:
        parts.append(f"Wikipedia says: {_simple_words(wiki['text'])}")
        srcs.append(f"Wikipedia ({wiki.get('url', '')})")
    if inv:
        parts.append(f"Investopedia page: {_simple_words(inv['text'])}")
        srcs.append(inv.get("url", "https://www.investopedia.com/"))
    return parts, srcs


def define(term: str) -> tuple[str, str]:
    """Return simple definition for term."""
    cleaned = term.strip()
    rows = retrieve(cleaned, 3)
    # always-web: Wikipedia + Investopedia live, in parallel (12-thread box)
    wiki, inv = _fetch_define_sources(cleaned)
    parts, srcs = _build_define_parts(rows, wiki, inv)
    if not parts:
        return (
            f"No definition found for '{cleaned}'. "
            f"Try RSI, P/E, ETF, inflation. {DISCLAIMER}",
            f"📖 {cleaned} — no source",
        )
    core = " ".join(parts[:2])
    gen = simplify_with_model(
        f"Q: What is {cleaned}?\nA: {core}\nQ: Say it super simply.\nA:",
    )
    simple = gen or (
        f"In super simple words: {core} "
        "Think school example — small ideas add up. "
        "Watch price + trend + news together."
    )
    txt = f"{simple}\n\nSources: {'; '.join(srcs[:3])}\n{DISCLAIMER}"
    return txt, f"📖 {cleaned} [Investopedia + Wikipedia + 5M]"


# ------------------------------------------------ symbol normalize
# Handles BTC/USD, btc-usd, bitcoin, company names, indices. No defaults.

_ALIASES = {
    "bitcoin": "BTC-USD", "btc": "BTC-USD", "btcusd": "BTC-USD",
    "ethereum": "ETH-USD", "eth": "ETH-USD",
    "nvidia": "NVDA", "apple": "AAPL", "tesla": "TSLA",
    "microsoft": "MSFT", "amazon": "AMZN", "meta": "META",
    "reliance": "RELIANCE.NS", "tcs": "TCS.NS",
    "infosys": "INFY.NS", "infy": "INFY.NS",
    "nifty": "^NSEI", "nifty 50": "^NSEI", "sensex": "^BSESN",
    "s&p": "^GSPC", "s&p 500": "^GSPC", "spx": "^GSPC", "sp500": "^GSPC",
    "nasdaq": "^NDX", "dow": "^DJI",
}
_STOP = {"WHY", "WHAT", "WHEN", "TODAY", "STOCK", "MARKET", "PRICE",
        "FALL", "FELL", "DROP", "RISE", "ROSE", "CRASH", "PUMP", "MONEY",
        "EARN", "EARNED", "MUCH", "MANY", "DOES", "HAVE", "THIS", "THAT",
        # finance abbreviations are words, not tickers
        "PE", "PB", "EPS", "ROE", "ROA", "ROI", "PEG", "RSI", "MACD",
        "SMA", "EMA", "ETF", "IPO", "GDP", "CPI", "TVM", "PFC", "YOY",
        "AI", "IT", "AM", "PM", "TV", "FM", "Q1", "Q2", "Q3", "Q4"}


def _alias_hit(text_lower: str) -> str | None:
    """Return alias symbol hit if present."""
    for name, sym in sorted(_ALIASES.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"(?<![a-z])" + re.escape(name) + r"(?![a-z])", text_lower):
            return sym
    return None


def _ticker_hit(text: str) -> str | None:
    """Return ticker-style symbol hit if present."""
    pattern = r"\b([A-Za-z]{1,6}(?:\.NS)?|\^[A-Za-z]{2,6}|BTC-USD|ETH-USD)\b"
    for match in re.finditer(pattern, text):
        raw, cand = match.group(1), match.group(1).upper()
        if cand in _ALIASES.values():
            return cand
        # generic tickers must be SHOUTED in the original (kills did/are/the)
        if (
            len(raw) >= 2
            and raw.isupper()
            and cand not in _STOP
            and re.fullmatch(r"[A-Z]{2,5}(\.NS)?", cand)
        ):
            return cand
    return None


def norm_symbol(text: str) -> str | None:
    """Find a market symbol in free text."""
    cleaned = text.strip()
    lowered = cleaned.lower()
    hit = _alias_hit(lowered)
    if hit is not None:
        return hit
    if re.search(r"btc\s*/\s*usd", lowered):
        return "BTC-USD"
    return _ticker_hit(cleaned)


def _offline_fundamentals_text(
    symbol: str, name: str, price_text: str,
) -> str:
    """Build offline fundamentals message."""
    return (
        f"{symbol} ({name}): no fundamentals right now (offline or "
        "non-equity like an index/crypto — those have no earnings). "
        f"Price {price_text}. {DISCLAIMER}"
    )


def _revenue_lines(
    fund: dict[str, Any], cli: Any, lines: list[str],
) -> None:
    """Append revenue lines."""
    if fund.get("rev_ttm"):
        lines.append(f"• Revenue TTM: {cli.fmt_money(fund['rev_ttm'])}")
    if fund.get("rev_qtr"):
        lines.append(
            f"• Revenue last quarter ({fund.get('qtr_date', '')}): "
            f"{cli.fmt_money(fund['rev_qtr'])} "
            "— this is the money it earned that quarter",
        )
    if fund.get("net_qtr") is not None:
        lines.append(
            f"• Net profit last quarter: {cli.fmt_money(fund['net_qtr'])}",
        )
    if fund.get("eps") is not None:
        lines.append(f"• EPS (profit per share): {fund['eps']:.2f}")


def _valuation_lines(fund: dict[str, Any], lines: list[str]) -> None:
    """Append valuation lines."""
    if fund.get("pe") is not None:
        lines.append(
            f"• P/E: {fund['pe']:.1f} (you pay this much per 1 of yearly profit)",
        )
    if fund.get("fpe") is not None:
        lines.append(f"• Forward P/E: {fund['fpe']:.1f}")
    if fund.get("peg") is not None:
        lines.append(f"• PEG: {fund['peg']:.2f} (<1 can mean cheap vs growth)")
    if fund.get("roe") is not None:
        lines.append(f"• ROE: {fund['roe'] * 100:.0f}% profit per owner money")
    if fund.get("margins") is not None:
        lines.append(
            f"• Profit margin: {fund['margins'] * 100:.1f}% "
            "of sales kept as profit",
        )


def _balance_lines(fund: dict[str, Any], cli: Any, lines: list[str]) -> None:
    """Append balance-sheet lines."""
    if fund.get("de") is not None:
        lines.append(f"• Debt/equity: {fund['de']:.0f}")
    if fund.get("cash"):
        lines.append(f"• Cash pile: {cli.fmt_money(fund['cash'])}")
    if fund.get("beta") is not None:
        lines.append(f"• Beta: {fund['beta']:.2f} (1 = moves with market)")
    if fund.get("div"):
        lines.append(f"• Dividend yield: {fund['div'] * 100:.2f}%/yr")
    if fund.get("target"):
        lines.append(f"• Analyst target: {cli.fmt_money(fund['target'])}")
    if fund.get("mcap"):
        lines.append(f"• Market cap: {cli.fmt_money(fund['mcap'])}")


def _long_extra_lines(symbol: str) -> list[str]:
    """Return extra lines for long output."""
    return [
        "• In simple words: revenue is sales collected, net profit is what "
        "owners keep, EPS splits profit per share, P/E shows how hopeful "
        "buyers are. Compare quarters, not just one.",
        f"• Chart it: python vaibhav_market_cli.py --history {symbol}",
    ]


def fundamentals_text(symbol: str, *, long: bool = False) -> str:
    """Return real fundamentals for symbol."""
    import vaibhav_market_cli as cli

    fund = cli.fetch_fundamentals(symbol)
    quote = cli.fetch_quote(symbol)
    name = fund.get("name") or quote.get("name") or symbol
    if not fund.get("ok"):
        price = cli.fmt_px(quote.get("price"), quote.get("currency", ""))
        return _offline_fundamentals_text(symbol, str(name), price)
    lines = [f"{symbol} ({name}) fundamentals [Yahoo Finance, free]:"]
    _revenue_lines(fund, cli, lines)
    _valuation_lines(fund, lines)
    _balance_lines(fund, cli, lines)
    if long:
        lines.extend(_long_extra_lines(symbol))
    body = "\n".join(lines)
    checked = verify_answer(
        body, {"rev": fund.get("rev_qtr"), "eps": fund.get("eps")},
    )
    body += f"\n{checked}"
    src = "FMP" if fund.get("src") == "fmp" else "Yahoo Finance"
    quote_src = quote.get("src", "")
    body += (
        f"\nSources: {src} fundamentals + quote ({quote_src}).\n{DISCLAIMER}"
    )
    return body


def _snapshot_line(quote: dict[str, Any]) -> str | None:
    """Build snapshot line for long move."""
    px = quote.get("price")
    chg = quote.get("chg")
    if px is None:
        return None
    return (
        f"• Snapshot: {px:,.2f} {quote.get('currency', '')} "
        f"({chg:+.2f}%)  O {quote.get('open', '—')} "
        f"H {quote.get('high', '—')} "
        f"L {quote.get('low', '—')} Vol {quote.get('vol', '—')}"
    )


def _range_line(quote: dict[str, Any]) -> str | None:
    """Build 52-week range line."""
    px = quote.get("price")
    yh = quote.get("yhigh")
    yl = quote.get("ylow")
    if not (px and yh and yl and yh != yl):
        return None
    pos = (px - yl) / (yh - yl) * 100
    return f"• 52-week range: {yl:,.0f}–{yh:,.0f}, now at {pos:.0f}% of the way up"


def _technicals_line(cli: Any, sym: str, px: Any) -> str | None:
    """Build technicals line for long move."""
    try:
        hist = cli.fetch_history(sym, "3mo")
        if hist is None or len(hist) <= 20:
            return None
        sma20 = cli.sma(hist, 20)[-1]
        rsi_val = cli.rsi(hist)
        ema12, ema26 = cli.ema(hist, 12), cli.ema(hist, 26)
        macd = float(ema12[-1] - ema26[-1])
    except Exception:
        return None
    else:
        trend = "above=uptrend" if px and px > sma20 else "below=weak"
        if rsi_val and rsi_val > 70:
            mood = "hot"
        elif rsi_val and rsi_val < 30:
            mood = "cold"
        else:
            mood = "neutral"
        push = "bull push" if macd > 0 else "bear push"
        return (
            f"• Technicals: SMA20 {sma20:,.2f} ({trend}), "
            f"RSI14 {rsi_val:.0f} ({mood}), "
            f"MACD {macd:+.2f} ({push})"
        )


def _quote_fundamentals_line(cli: Any, sym: str) -> str | None:
    """Build fundamentals bits line."""
    try:
        fund = cli.fetch_fundamentals(sym)
        if not fund.get("ok"):
            return None
        bits: list[str] = []
        if fund.get("rev_qtr"):
            bits.append(f"last-qtr revenue {cli.fmt_money(fund['rev_qtr'])}")
        if fund.get("net_qtr") is not None:
            bits.append(f"net profit {cli.fmt_money(fund['net_qtr'])}")
        if fund.get("pe"):
            bits.append(f"P/E {fund['pe']:.1f}")
    except Exception:
        return None
    else:
        if not bits:
            return None
        return "• Fundamentals: " + ", ".join(bits)


def _headlines_block(heads: list[dict[str, Any]]) -> list[str]:
    """Build headlines block lines."""
    if not heads:
        return []
    out = ["• Why it moved — latest headlines:"]
    out.extend(
        f"  {i}. {h['title'][:140]} [{h.get('src', '')}]"
        for i, h in enumerate(heads[:3], 1)
    )
    return out


def _long_move(cli: Any, sym: str, quote: dict[str, Any], heads: list[Any]) -> str:
    """Return super-long detail for symbol."""
    out = [f"{sym} ({quote.get('name', sym)}) — long detail [live, free data]:"]
    snap = _snapshot_line(quote)
    if snap is not None:
        out.append(snap)
    rng = _range_line(quote)
    if rng is not None:
        out.append(rng)
    tech = _technicals_line(cli, sym, quote.get("price"))
    if tech is not None:
        out.append(tech)
    fund_line = _quote_fundamentals_line(cli, sym)
    if fund_line is not None:
        out.append(fund_line)
    out.extend(_headlines_block(heads))
    out.append(
        "• In simple words: price falls when sellers outnumber buyers. "
        "Headlines spark it, volume confirms it, trend shows how strong it is.",
    )
    out.append(f"• See it: python vaibhav_market_cli.py --history {sym} --period 3mo")
    body = "\n".join(out)
    conf = confidence(body)
    checked = verify_answer(
        body, {"price": quote.get("price"), "chg": quote.get("chg")},
    )
    body += f"\n{checked}{(' · ' + conf) if conf else ''}"
    body += (
        f"\nSources: live quote ({quote.get('src', '')}), Yahoo headlines, "
        f"Yahoo fundamentals, NumPy technicals.\n{DISCLAIMER}"
    )
    return body


# ------------------------------------------------ session context
# Remembers this chat: last symbol, last topic, Q&A trail. Follow-ups like
# "and its PE?" or "why?" resolve against it. Lives per process.

_CTX: dict[str, Any] = {"history": [], "symbol": None, "topic": None}


def _remember(q: str, title: str, symbol: str | None) -> None:
    """Remember question and symbol."""
    if symbol:
        _CTX["symbol"] = symbol
    _CTX["topic"] = title
    history: Any = _CTX.get("history", [])
    history.append((q, title))
    _CTX["history"] = history[-8:]


def _resolve_followup(q: str, context_symbol: str | None) -> str | None:
    """Resolve follow-up symbol from context."""
    direct = norm_symbol(q)
    if direct or context_symbol or _CTX.get("symbol"):
        return direct or context_symbol or _CTX.get("symbol")
    if re.search(
        r"\b(its|it|that|this one|same)\b", q, re.IGNORECASE,
    ) and _CTX.get("symbol"):
        return _CTX.get("symbol")
    return None


def _is_fundamentals_intent(q: str, sym0: str | None) -> bool:
    """Check whether question wants fundamentals."""
    if re.search(r"\bfund[a-z]*\b", q, re.IGNORECASE):
        return True
    return bool(
        sym0
        and re.search(
            r"\b(earn[a-z]*|revenue|profit|quarter|eps|p/?e\b|q[1-4]|results?)\b",
            q,
            re.IGNORECASE,
        ),
    )


def _answer_fundamentals(sym0: str | None) -> tuple[str, str]:
    """Answer fundamentals intent."""
    if not sym0:
        return (
            "To show fundamentals I need a company, e.g.:\n"
            '• python vaibhav_market_cli.py --ask "fundamentals of NVDA"\n'
            '• python vaibhav_market_cli.py --ask '
            '"how much did AAPL earn last quarter"\n'
            "In simple words meanwhile: revenue = sales collected, "
            "net profit = "
            f"what owners keep, EPS = profit per share. {DISCLAIMER}",
            "🏦 fundamentals — which company?",
        )
    return fundamentals_text(sym0, long=True), f"🏦 {sym0} fundamentals [Yahoo free]"


def _answer_define(q: str) -> tuple[str, str] | None:
    """Answer definition shortcut if matched."""
    match = re.search(
        r"(?:what is|what are|define|meaning of)\s+([a-zA-Z/ \-]+?)(?:\?|$)",
        q,
        re.IGNORECASE,
    )
    if not match:
        return None
    text, title = define(match.group(1).strip(" ?"))
    return text, title


def _answer_symbol_move(
    cli: Any, sym: str, q: str, *, long: bool,
) -> tuple[str, str]:
    """Answer symbol move question."""
    quote = cli.fetch_quote(sym)
    heads = cli.fetch_news(sym if not sym.startswith("^") else None, 3)
    px = quote.get("price")
    chg = quote.get("chg")
    line = (
        f"{sym} is {px:,.2f} {quote.get('currency', '')} "
        f"({chg:+.2f}%) now. "
        if px and chg is not None
        else f"{sym} quote checked. "
    )
    why = " ".join(h["title"][:110] for h in heads[:2])
    if long:
        return _long_move(cli, sym, quote, heads), f"🤖 {sym} deep detail [live]"
    gen = simplify_with_model(f"Q: {q}\nFacts: {line} News: {why}\nA:")
    simple = gen or (
        f"{line}In simple words: price moves when sellers outnumber buyers. "
        f"Latest news: {why or 'no single headline'} "
        "Check trend, volume and one-week chart before deciding."
    )
    src = f"Live quote ({quote.get('src', '')}) + Yahoo headlines"
    return (
        f"{simple[:900]}\n\nSources: {src}; Investopedia glossary\n{DISCLAIMER}",
        f"🤖 {sym} now [live + 5M]",
    )


def _answer_market_wide(cli: Any) -> tuple[str, str]:
    """Answer market-wide move question."""
    heads = cli.fetch_news(None, 4)
    idx = [cli.fetch_quote(s) for s in ("^GSPC", "^NSEI", "BTC-USD")]
    snap = " · ".join(
        f"{d['symbol']} {d['chg']:+.2f}%"
        if d.get("chg") is not None
        else f"{d['symbol']} —"
        for d in idx
    )
    why = "\n".join(
        f"• {h['title'][:130]} [{h.get('src', '')}]" for h in heads[:3]
    )
    return (
        f"Market now: {snap}.\n{why}\nIn simple words: when fear news "
        "(rates, oil, AI doubts) spreads, sellers outnumber buyers across "
        "many stocks at once. For one company, ask with its symbol, e.g. "
        '"why did BTC-USD fall today in long detail".\n\n'
        f"Sources: live quotes + Yahoo headlines.\n{DISCLAIMER}",
        "🤖 market now [live]",
    )


def _is_move_intent(q: str, sym: str | None) -> bool:
    """Check whether question is a move question."""
    if sym:
        return True
    keywords = (
        "fall",
        "fell",
        "drop",
        "dropped",
        "rise",
        "rose",
        "pump",
        "crash",
        "today",
    )
    return any(w in q.lower() for w in keywords)


def _answer_move(
    q: str, sym: str | None, context_symbol: str | None, *, long: bool,
) -> tuple[str, str] | None:
    """Answer move or market question if matched."""
    resolved = sym or (context_symbol or None)
    if not _is_move_intent(q, resolved):
        return None
    try:
        import vaibhav_market_cli as cli

        if resolved:
            result = _answer_symbol_move(cli, resolved, q, long=long)
        else:
            result = _answer_market_wide(cli)
    except Exception:
        return None
    else:
        return result


def _answer_fallback(q: str) -> tuple[str, str]:
    """Answer with glossary fallback."""
    rows = retrieve(q, 3)
    ctx = " ".join(r["text"][:250] for r in rows)
    # market/news context: top headlines
    news_ctx = ""
    try:
        import vaibhav_market_cli as cli

        heads = cli.fetch_news(None, 3)
        news_ctx = " Headlines now: " + " | ".join(
            h["title"][:90] for h in heads
        )
    except Exception:
        pass
    gen = simplify_with_model(f"Q: {q}\nFacts: {ctx[:400]}{news_ctx[:200]}\nA:")
    base = gen or (
        f"Here is the simple version. {ctx[:400]} {news_ctx[:200]}".strip()
        or "I could not find that. Try asking to define RSI, ETF, "
        "inflation, or explain headlines."
    )
    srcs = ["Investopedia glossary (local)"]
    if rows:
        srcs.append(f"top match: {rows[0]['term']}")
    srcs += ["Wikipedia (for definitions)", "Yahoo Finance headlines (for news)"]
    return (
        f"{base[:700]}\n\nSources: {'; '.join(srcs)}\n{DISCLAIMER}",
        "🤖 answer [RAG + 5M + web]",
    )


def _answer_inner(
    question: str, context_symbol: str | None = None,
) -> tuple[str, str]:
    """Answer question with routing."""
    q = question.strip()
    long = bool(
        re.search(
            r"\b(long|detail|detailed|explain more|in depth)\b",
            q,
            re.IGNORECASE,
        ),
    )
    sym0 = _resolve_followup(q, context_symbol)
    # 1) fundamentals intent FIRST (typo-tolerant: fundemnetal, fundametals…)
    if _is_fundamentals_intent(q, sym0):
        return _answer_fundamentals(sym0)
    # 2) definition shortcut stays exact
    defined = _answer_define(q)
    if defined is not None:
        return defined
    # 3) why-move question: symbol-aware, else market-wide (never NVDA default)
    moved = _answer_move(q, norm_symbol(q), context_symbol, long=long)
    if moved is not None:
        return moved
    return _answer_fallback(q)


def answer(
    question: str, context_symbol: str | None = None,
) -> tuple[str, str]:
    """Wrap answer with memory for follow-ups."""
    txt, ttl = _answer_inner(question, context_symbol)
    match = re.search(r"🏦 (\S+)|🤖 (\S+)", ttl)
    symbol = match.group(1) or match.group(2) if match else None
    _remember(question, ttl, symbol)
    return txt, ttl


def verify_answer(text: str, facts: dict[str, Any]) -> str:
    """Check that numbers appear in text.

    facts: {"price": 211.01, "chg": -3.34, "rev_qtr": 9.6e10, ...}.
    Returns '✓ numbers match live data' or lists mismatches. Own code.
    """
    bad: list[str] = []
    for key, val in facts.items():
        if val is None:
            continue
        try:
            num = float(val)  # type: ignore[arg-type]
        except Exception:
            continue
        flat = text.replace(",", "")
        cands = [
            f"{num:,.2f}".rstrip("0").rstrip(".").split(".")[0].replace(",", ""),
        ]
        magnitude = abs(num)
        for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if magnitude >= div:
                short = f"{num / div:.2f}".rstrip("0").rstrip(".") + suf
                cands.append(short)
                cands.append(short.replace(".", ""))
                break
        if not any(c and c in flat for c in cands):
            bad.append(key)
    if not bad:
        return "✓ numbers match live data"
    return "⚠ could not confirm in text: " + ", ".join(bad) + " — check sources below"


def _load_checker_model() -> tuple[Any, Any] | None:
    """Load 300k checker model."""
    import torch

    from vaibhav_ai_train import GPTConfig, TinyGPT

    if not CHECK_PATH.exists():
        return None
    with VOCAB_PATH.open(encoding="utf-8") as handle:
        vocab_data: Any = json.load(handle)
    ck: Any = torch.load(str(CHECK_PATH), map_location="cpu")
    cfg_data: Any = ck.get("cfg", {})
    cfg = GPTConfig(
        vocab=len(vocab_data["itos"]),
        block=cfg_data.get("block", 96),
        n_layer=cfg_data.get("n_layer", 4),
        n_head=cfg_data.get("n_head", 4),
        n_embd=cfg_data.get("n_embd", 80),
    )
    model: Any = TinyGPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(requires_grad=False)
    return model, (vocab_data["itos"], vocab_data["stoi"])


def _score_fluency(text: str, model: Any, stoi: dict[str, int]) -> str:
    """Score fluency with perplexity."""
    import torch

    ids = torch.tensor([[stoi.get(c, 0) for c in text[-200:]]])
    with torch.inference_mode():
        _, loss = model(ids[:, :-1], ids[:, 1:])
    ppl = float(torch.exp(loss).item())
    if ppl < 6:
        return "✓ checker: fluent"
    if ppl < 12:
        return "~ checker: okay, verify numbers"
    return "⚠ checker: low confidence — verify with sources"


def _ensure_checker() -> Any | None:
    """Ensure checker model is loaded."""
    if globals().get("_CHECK") is not None:
        return globals().get("_CHECK")
    try:
        loaded = _load_checker_model()
    except Exception:
        return None
    else:
        if loaded is None:
            return None
        globals()["_CHECK"] = loaded
        return loaded


def confidence(text: str) -> str:
    """Score fluency with 300k checker."""
    if os.environ.get("VAIBHAV_NO_MODEL") == "1":
        return ""
    if not CHECK_PATH.exists():
        return ""
    cached = _ensure_checker()
    if cached is None:
        return ""
    try:
        model, _itos, stoi = cached
        result = _score_fluency(text, model, stoi)
    except Exception:
        return ""
    else:
        return result


def _explain_single(
    headline: dict[str, Any],
) -> str:
    """Explain single headline simply."""
    title = str(headline.get("title", ""))[:140]
    rows = retrieve(title, 1)
    hint = f" Related idea: {rows[0]['term']}." if rows else ""
    gen = simplify_with_model(f"Q: Explain simply: {title}\nA:")
    simple = gen or (
        f"It means buyers and sellers reacted to news and mood.{hint} "
        "Check trend and volume before deciding."
    )
    return f"• {title}\n  → {simple[:220]} [{headline.get('src', '')}]"


def explain_news(headlines: list[dict[str, Any]]) -> str:
    """Explain headlines in simple words."""
    if not headlines:
        return f"No headlines right now. {DISCLAIMER}"
    lines = [_explain_single(item) for item in headlines[:5]]
    return (
        "\n".join(lines)
        + "\n\nSources: Yahoo Finance RSS + Investopedia glossary + 5M model.\n"
        f"{DISCLAIMER}"
    )


def _chat_intro(console: Any) -> None:
    """Print chat intro banner."""
    console.print(
        "[bold magenta]🤖 Vaibhav chat — asks remember context "
        "(/symbol X · /news · /quit)[/]",
    )


def _chat_news(console: Any, panel: Any, sym: str | None) -> None:
    """Print explained news block."""
    import vaibhav_market_cli as cli

    news = cli.fetch_news(sym, 5) if sym else cli.fetch_news(None, 5)
    console.print(panel(explain_news(news), title="📰 explained"))


def _chat_step(
    raw: str, sym: str | None, console: Any, panel: Any,
) -> tuple[str | None, bool]:
    """Process single chat line."""
    text = raw.strip()
    if not text:
        return sym, True
    lowered = text.lower()
    if lowered in ("/quit", "/q", "quit", "exit"):
        return sym, False
    if lowered.startswith("/symbol"):
        return norm_symbol(text[7:]) or None, True
    if lowered == "/news":
        _chat_news(console, panel, sym)
        return sym, True
    if lowered in ("/help", "help", "?"):
        console.print(
            "[dim]/symbol X — set context · /news — explain headlines · "
            "/quit — leave. Follow-ups like 'and its PE?' use context.[/]",
        )
        return sym, True
    body, title = answer(text, context_symbol=sym)
    console.print(panel(body, title=title, border_style="green"))
    return sym, True


def chat_loop(symbol: str | None = None) -> None:
    """Run interactive chat with memory."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    _chat_intro(console)
    sym = norm_symbol(symbol or "") if symbol else None
    while True:
        try:
            raw = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        updated, keep = _chat_step(raw, sym, console, Panel)
        if raw.strip().lower().startswith("/symbol") and keep:
            sym = updated
            console.print(f"[cyan]context symbol: {sym or 'cleared'}[/]")
            continue
        sym = updated
        if not keep:
            break
    console.print("[dim]bye — Vaibhav/GPL-v3[/]")


def _run_cli() -> None:
    """Run command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--define", default=None)
    parser.add_argument("--ask", default=None)
    parser.add_argument(
        "--symbol", default=None, help="context symbol for --ask/--chat",
    )
    parser.add_argument(
        "--chat", action="store_true", help="interactive chat with memory",
    )
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    if args.define:
        text, title = define(args.define)
        print(f"== {title} ==\n{text}")
    elif args.ask:
        ctx = norm_symbol(args.symbol) if args.symbol else None
        text, title = answer(args.ask, context_symbol=ctx)
        print(f"== {title} ==\n{text}")
    elif args.chat:
        chat_loop(args.symbol)
    elif args.explain:
        import vaibhav_market_cli as cli

        print(explain_news(cli.fetch_news(None, 5)))
    else:
        parser.print_help()


if __name__ == "__main__":
    _run_cli()
