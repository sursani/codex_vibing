"""Shared library for stock financial analysis.

This module exposes asynchronous helper functions that are reused by both
the command‑line interface (``stock_analysis_app.py``) and the Streamlit UI
(``streamlit_stock_app.py``).

The original logic from *stock_analysis_app.py* has been refactored here so
that there is a single source of truth for:

1. Fetching the latest price using an OpenAI Tools agent (web search).
2. Fetching trailing/forward P‑E ratios via *yfinance*.
3. Running a large‑language‑model agent to produce an overall analysis.

Any new front‑end (CLI, Streamlit, etc.) can therefore simply import and call
these functions without duplicating code.
"""

from __future__ import annotations

import os
import asyncio
from typing import Tuple, Optional

import yfinance as yf

from pydantic import BaseModel

# OpenAI Agents SDK imports
from agents import Agent, Runner, WebSearchTool
from agents.tool import function_tool
from agents.model_settings import ModelSettings


# ---------------------------------------------------------------------------
# Pydantic result schemas (kept identical to the originals)
# ---------------------------------------------------------------------------


class PriceResult(BaseModel):
    ticker: str
    price: float


class PERatiosResult(BaseModel):
    ticker: str
    trailingPE: Optional[float] = None
    forwardPE: Optional[float] = None


class AnalysisResult(BaseModel):
    ticker: str
    price: float
    trailingPE: Optional[float]
    forwardPE: Optional[float]
    analysis: str


# ---------------------------------------------------------------------------
# Agents & tools
# ---------------------------------------------------------------------------


# Agent that looks up the latest price with the WebSearchTool provided by the
# agents SDK. It returns a ``PriceResult`` JSON object.
_PRICE_AGENT_PROMPT = (
    "You are an assistant that fetches the latest stock price for a given "
    "ticker symbol. Use the provided web search tool to look up the current "
    "price and return JSON: {\"ticker\": TICKER, \"price\": PRICE}."
)

_price_agent = Agent(
    name="PriceAgent",
    instructions=_PRICE_AGENT_PROMPT,
    tools=[WebSearchTool()],
    model_settings=ModelSettings(tool_choice="required"),
    output_type=PriceResult,
)


# ---------------------------------------------------------------------------
# yfinance helper wrapped as an agents SDK function tool so it can be called by
# the LLM if needed. We will also expose it for direct synchronous use.
# ---------------------------------------------------------------------------


@function_tool
def get_pe_ratios(ticker: str) -> PERatiosResult:  # noqa: D401
    """Return trailing and forward P/E ratios using *yfinance*.

    The `yfinance.Ticker.info` field sometimes lacks either ratio, so values
    may be *None*.
    """

    t = yf.Ticker(ticker)
    info = t.info or {}
    return PERatiosResult(
        ticker=ticker,
        trailingPE=info.get("trailingPE"),
        forwardPE=info.get("forwardPE"),
    )


# A lightweight agent whose sole purpose is to call ``get_pe_ratios`` and
# return the tool output as structured JSON.

def _build_pe_agent() -> Agent:
    return Agent(
        name="PERatioAgent",
        instructions="You are an assistant that returns P/E ratios for a given ticker.",
        tools=[get_pe_ratios],
        model_settings=ModelSettings(tool_choice="required"),
        output_type=PERatiosResult,
    )


# ---------------------------------------------------------------------------
# Public asynchronous helper functions
# ---------------------------------------------------------------------------


async def fetch_latest_price(ticker: str) -> PriceResult:
    """Return the latest price for *ticker* using the PriceAgent."""

    run = await Runner.run(_price_agent, ticker)
    return run.final_output  # type: ignore[return-value]


async def fetch_pe_ratios(ticker: str) -> PERatiosResult:
    """Return trailing/forward P‑E using PERatioAgent."""

    pe_agent = _build_pe_agent()
    run = await Runner.run(pe_agent, ticker)
    return run.final_output  # type: ignore[return-value]


async def run_llm_analysis(
    ticker: str,
    price: float,
    trailing_pe: Optional[float],
    forward_pe: Optional[float],
    model: str | None = None,
) -> AnalysisResult:
    """Invoke the analyst LLM to comment on the company outlook."""

    analysis_prompt = (
        "You are a financial analyst. Given the following data, provide a detailed "
        "analysis of the company's outlook:\n"
        "Ticker: {ticker}\n"
        "Latest Price: {price}\n"
        "Trailing P/E: {trailing_pe}\n"
        "Forward P/E: {forward_pe}\n"
        "Return JSON matching the schema: {{\"ticker\": str, \"price\": float, "
        "\"trailingPE\": float|null, \"forwardPE\": float|null, \"analysis\": str}}."
    ).format(
        ticker=ticker,
        price=price,
        trailing_pe=trailing_pe,
        forward_pe=forward_pe,
    )

    analysis_agent = Agent(
        name="AnalysisAgent",
        instructions=analysis_prompt,
        tools=[],
        output_type=AnalysisResult,
        model=model,
    )

    run = await Runner.run(analysis_agent, "")
    return run.final_output  # type: ignore[return-value]


async def analyze_ticker(
    ticker: str,
    model: str | None = None,
) -> Tuple[PriceResult, PERatiosResult, AnalysisResult]:
    """High‑level convenience coroutine: price → P‑E → LLM analysis."""

    # Run price and P‑E agents concurrently for speed
    price_task = asyncio.create_task(fetch_latest_price(ticker))
    pe_task = asyncio.create_task(fetch_pe_ratios(ticker))

    price_res, pe_res = await asyncio.gather(price_task, pe_task)

    analysis_res = await run_llm_analysis(
        ticker=ticker,
        price=price_res.price,
        trailing_pe=pe_res.trailingPE,
        forward_pe=pe_res.forwardPE,
        model=model,
    )

    return price_res, pe_res, analysis_res


# ---------------------------------------------------------------------------
# Non‑async helpers (for Streamlit convenience)
# ---------------------------------------------------------------------------


def fetch_price_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
):
    """Return a *pandas* DataFrame of historical OHLCV data via yfinance."""

    return yf.Ticker(ticker).history(period=period, interval=interval)


# ---------------------------------------------------------------------------
# Sanity check when executed directly
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    """CLI smoke test: ``python stock_analysis_lib.py AAPL``."""

    import sys, json  # noqa: E402

    if len(sys.argv) < 2:
        print("Usage: python stock_analysis_lib.py TICKER")
        raise SystemExit(1)

    ticker_arg = sys.argv[1]

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        raise SystemExit(1)

    async def _demo() -> None:  # noqa: D401
        price, pe, analysis = await analyze_ticker(ticker_arg)
        print("Price:", price.model_dump())
        print("PE:", pe.model_dump())
        print(json.dumps(analysis.model_dump(), indent=2))

    asyncio.run(_demo())
