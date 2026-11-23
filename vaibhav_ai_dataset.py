# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026 Vaibhav — final power over this project: Vaibhav.
# License default: GPL-3.0-only, NEVER MIT. Ask Vaibhav before public/GitHub.
"""Build finance dataset for Vaibhav 5M model. CPU-only, venv-only.

Sources (free, credible, attributed at inference):
 1. HuggingFace infCapital/investopedia_terms_en (6.3k Investopedia terms)
 2. Curated fallback glossary (~250 core terms, hand-simplified)
 3. Live Yahoo headlines -> simple-explainer pairs (templates, no copying)
Output: data/train.txt, data/val.txt, data/vocab.json, data/glossary.json
Run: source ~/ml_env/bin/activate && python vaibhav_ai_dataset.py [--live-news 200]
"""
from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
random.seed(7)

CURATED: list[tuple[str, str]] = [
    (
        "RSI",
        "Relative Strength Index. A 0-100 meter for speed of price "
        "moves. Above 70 often means overbought and tired buyers. Below "
        "30 often means oversold and tired sellers. It does not "
        "predict, it describes momentum.",
    ),
    (
        "P/E ratio",
        "Price to Earnings. How much you pay for 1 rupee/dollar of "
        "yearly profit. P/E 20 means you pay 20 for every 1 of profit. "
        "High can mean high hopes. Low can mean cheap or troubled.",
    ),
    (
        "Market cap",
        "Total value of all shares. Price times number of shares. Big "
        "cap means big stable company. Small cap means small, faster "
        "but riskier.",
    ),
    (
        "Dividend",
        "Cash a company shares with owners from profit. Like a "
        "thank-you payment. High dividend can mean steady cash, but "
        "also slow growth.",
    ),
    (
        "SMA",
        "Simple Moving Average. Average price of last N days. Smooths "
        "noise so you see the trend. Price above SMA often means "
        "uptrend.",
    ),
    (
        "EMA",
        "Exponential Moving Average. Like SMA but listens more to "
        "recent days. Faster to react than SMA.",
    ),
    (
        "MACD",
        "MACD. Difference of two EMAs (12 and 26). Above zero often "
        "means bull push. Below zero often means bear push.",
    ),
    (
        "Bull market",
        "A long rise where buyers are strong and confident. Prices "
        "climb like a bull charging up.",
    ),
    (
        "Bear market",
        "A long fall where sellers are strong and fear rules. Prices "
        "slide like a bear swiping down.",
    ),
    (
        "Volatility",
        "How much price jumps around. High volatility means big swings "
        "and big risk. Low means calm.",
    ),
    (
        "Volume",
        "How many shares changed hands. High volume means strong "
        "interest. Low means quiet.",
    ),
    (
        "52-week high/low",
        "Highest and lowest price in the last year. Shows the full "
        "playing field for the year.",
    ),
    (
        "Bid and ask",
        "Bid is the highest price a buyer will pay. Ask is the lowest "
        "price a seller will take. The gap is the spread.",
    ),
    (
        "Spread",
        "Gap between bid and ask. Small spread means easy cheap "
        "trading. Big spread means costly.",
    ),
    (
        "ETF",
        "Exchange Traded Fund. A basket of many stocks you buy as one. "
        "Spreads risk across many companies.",
    ),
    (
        "Index",
        "A group score like Nifty 50 or S&P 500. Shows how the whole "
        "market family is doing.",
    ),
    (
        "Nifty 50",
        "India's top 50 big companies on NSE grouped into one score. "
        "When Nifty rises, big India is rising.",
    ),
    (
        "S&P 500",
        "America's top 500 big companies grouped into one score. The "
        "world's most watched market score.",
    ),
    (
        "Inflation",
        "Prices rising over time so money buys less. Small inflation is "
        "normal. High inflation hurts savings.",
    ),
    (
        "Interest rate",
        "Cost of borrowing money. High rates slow business. Low rates "
        "speed it up.",
    ),
    (
        "Revenue",
        "Total money a company collects from sales before any costs. "
        "The top line.",
    ),
    (
        "Profit",
        "Money left after all costs. Revenue minus costs. The bottom "
        "line owners keep.",
    ),
    (
        "EPS",
        "Earnings Per Share. Profit divided by number of shares. More "
        "EPS usually means healthier per-share earning.",
    ),
    (
        "Balance sheet",
        "A photo of what a company owns (assets), owes (debt), and "
        "owners keep (equity) on one day.",
    ),
    (
        "Beta",
        "How much a stock jumps vs the market. Beta 1 moves with "
        "market. Beta 2 jumps double. Below 1 is calmer.",
    ),
    (
        "Short selling",
        "Borrowing shares to sell now, hoping to buy back cheaper "
        "later. Profit if price falls. Risky because rises hurt.",
    ),
    (
        "Option",
        "A paid choice to buy or sell later at a fixed price. You pay a "
        "small fee for the chance, not the duty.",
    ),
    (
        "Bond",
        "A loan you give to a company or government. They pay you "
        "interest and return money later. Calmer than stocks.",
    ),
    (
        "Portfolio",
        "Your full set of investments together. Diversified means many "
        "types so one fall does not sink you.",
    ),
    (
        "Diversification",
        "Not putting all eggs in one basket. Spread across many stocks "
        "and types to lower risk.",
    ),
    (
        "Risk",
        "Chance of losing money or getting less than hoped. Higher "
        "possible reward usually needs higher risk.",
    ),
]


def _clean(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    return cleaned[:600]


def load_hf_terms() -> list[tuple[str, str]]:
    """Load terms from HuggingFace."""
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "infCapital/investopedia_terms_en", split="train",
        )
        out: list[tuple[str, str]] = []
        for row in dataset:
            name = _clean(str(row.get("name", "")))
            text = _clean(str(row.get("text", "")))
            if len(name) > 1 and len(text) > 40:
                out.append((name, text))
    except Exception as err:
        print(
            f"HF dataset offline ({err}), using curated fallback",
            flush=True,
        )
        return []
    else:
        print(
            f"HF investopedia_terms_en: {len(out)} terms", flush=True,
        )
        return out


def simplify(term: str, definition: str) -> str:
    """Explain a term simply."""
    short = definition[:400]
    return (
        f"{term} means this in simple words. {short} "
        "Think of it like a school example: small steps add up. "
        "In the market, watch price, trend and news together "
        "before deciding."
    )


def _add_news_pairs(pairs: list[str], live_news: int) -> None:
    """Add news explainer pairs."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        cli = importlib.import_module("vaibhav_market_cli")

        heads = cli.fetch_news(None, min(30, live_news))
        for item in heads:
            title = _clean(str(item.get("title", "")))[:160]
            if len(title) > 20:
                pairs.append(
                    f"Q: Explain this headline simply: {title}\n"
                    "A: This headline says the market moved "
                    "because of news and mood. "
                    "In simple words: prices move when many "
                    "buyers or sellers act together. "
                    "Check the price trend, volume and company "
                    "news before deciding. "
                    "This is learning, not financial advice.\n",
                )
        print(f"news pairs: {len(heads)}", flush=True)
    except Exception as err:
        print(f"live news skip ({err})", flush=True)


def _generic_pairs() -> list[str]:
    """Return generic market Q&A pairs."""
    return [
        (
            "Q: Why did my stock fall today?\n"
            "A: Stocks fall when more people sell than buy. Common "
            "reasons: weak results, bad news, high fear, or full market "
            "falling. Check news, volume and trend. One day does not "
            "tell the full story.\n"
        ),
        (
            "Q: Should I buy now?\n"
            "A: I cannot tell you to buy or sell. I can teach: check "
            "trend (price above SMA 20?), strength (RSI near 70 means "
            "hot, near 30 means cold), news reason, and your own risk "
            "limit. Learning first, decision yours.\n"
        ),
        (
            "Q: What is a trend?\n"
            "A: Trend is the general direction. Up trend means higher "
            "highs over weeks. Down trend means lower lows. Sideways "
            "means waiting. Sparklines and SMAs show it simply.\n"
        ),
        (
            "Q: What moves the market?\n"
            "A: Earnings, interest rates, inflation news, oil, wars, "
            "and crowd mood. Big buyers move prices most. News starts "
            "it, trend confirms it.\n"
        ),
    ]


def build(live_news: int = 0) -> None:
    """Build dataset files."""
    DATA.mkdir(parents=True, exist_ok=True)
    terms = load_hf_terms()
    seen = {term.lower() for term, _ in terms}
    for term, defi in CURATED:
        if term.lower() not in seen:
            terms.append((term, defi))
    print(f"total glossary: {len(terms)}", flush=True)
    source = (
        "Investopedia (via HF infCapital/"
        "investopedia_terms_en or curated)"
    )
    gloss = [
        {"term": term, "text": text, "source": source}
        for term, text in terms
    ]
    with (DATA / "glossary.json").open("w", encoding="utf-8") as handle:
        json.dump(gloss, handle, indent=1)
    pairs: list[str] = []
    for term, defi in terms:
        pairs.append(f"Q: What is {term}?\nA: {simplify(term, defi)}\n")
        pairs.append(
            f"Q: Define {term} simply.\nA: {simplify(term, defi)}\n",
        )
    if live_news > 0:
        _add_news_pairs(pairs, live_news)
    pairs += _generic_pairs()
    random.shuffle(pairs)
    count = len(pairs)
    val_count = max(20, count // 10)
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]
    with (DATA / "train.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(train_pairs))
    with (DATA / "val.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(val_pairs))
    text = "\n".join(train_pairs)
    chars = sorted(set(text))
    stoi = {char: idx for idx, char in enumerate(chars)}
    with (DATA / "vocab.json").open("w", encoding="utf-8") as handle:
        json.dump({"itos": chars, "stoi": stoi}, handle)
    print(
        f"pairs train={len(train_pairs)} val={len(val_pairs)} "
        f"vocab={len(chars)} chars",
        flush=True,
    )
    print(
        f"wrote {DATA}/train.txt val.txt vocab.json glossary.json",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-news", type=int, default=0)
    args = parser.parse_args()
    build(args.live_news)
