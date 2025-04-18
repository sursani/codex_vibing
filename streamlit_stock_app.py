"""Streamlit dashboard for AI‑powered stock analysis."""

from __future__ import annotations

import asyncio
import datetime as _dt
import os
from pathlib import Path

# Ensure environment variables from a local .env file are loaded before we
# access OPENAI_API_KEY. This mirrors the behaviour of the CLI script.
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)
except ImportError:  # pragma: no cover
    pass

import plotly.express as px
import streamlit as st

from stock_analysis_lib import (
    analyze_ticker,
    fetch_price_history,
)


# ---------------------------------------------------------------------------
# Page configuration & global settings
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="AI Stock Analyzer",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar – user inputs
# ---------------------------------------------------------------------------


with st.sidebar:
    st.header("🔍 Query")

    default_ticker = "AAPL"
    ticker = st.text_input("Ticker symbol", value=default_ticker).upper().strip()

    st.markdown("---")

    st.subheader("Price history window")
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "Max": "max",
    }
    period_label = st.selectbox("Period", list(period_options.keys()), index=3)
    period = period_options[period_label]

    interval = st.selectbox(
        "Interval",
        [
            "1d",
            "1wk",
            "1mo",
        ],
        index=0,
    )

    st.markdown("---")

    model_override = st.text_input("LLM model override (optional)") or None

    analyze_button = st.button("Analyze", type="primary")


# ---------------------------------------------------------------------------
# Main application logic
# ---------------------------------------------------------------------------


def _require_api_key() -> bool:
    """Return *True* if OPENAI_API_KEY exists else render error and return *False*."""

    if os.getenv("OPENAI_API_KEY"):
        return True

    st.error("OPENAI_API_KEY environment variable is not set. Please add it to your environment and restart.")
    return False


def _run_async(coro):
    """Helper to run an *async* coroutine inside Streamlit’s sync context."""

    return asyncio.run(coro)


if analyze_button and _require_api_key():
    # ---------------------------------------------------------------------
    # Fetch data & run agents with a nice spinner
    # ---------------------------------------------------------------------

    with st.spinner("Running agents and fetching data…"):
        try:
            price_res, pe_res, analysis_res = _run_async(analyze_ticker(ticker, model=model_override))
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
            st.stop()

    # ------------------------------------------------------------------
    # Display key metrics
    # ------------------------------------------------------------------

    col_price, col_trailing, col_forward = st.columns(3)

    col_price.metric("Latest Price", f"${price_res.price:,.2f}")
    trailing_val = "N/A" if pe_res.trailingPE is None else f"{pe_res.trailingPE:,.2f}"
    forward_val = "N/A" if pe_res.forwardPE is None else f"{pe_res.forwardPE:,.2f}"

    col_trailing.metric("Trailing P/E", trailing_val)
    col_forward.metric("Forward P/E", forward_val)

    st.markdown("---")

    # ------------------------------------------------------------------
    # Price history chart
    # ------------------------------------------------------------------

    hist_df = fetch_price_history(ticker, period=period, interval=interval)

    if hist_df.empty:
        st.warning("No historical data available for this ticker.")
    else:
        hist_df = hist_df.reset_index()
        fig = px.line(
            hist_df,
            x="Date",
            y="Close",
            title=f"{ticker} Price History ({period_label})",
            labels={"Close": "Price (USD)"},
        )
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # LLM analysis text
    # ------------------------------------------------------------------

    st.subheader("Analyst commentary 🧠")
    st.write(analysis_res.analysis)

    st.markdown("---")

    with st.expander("Raw JSON output"):
        st.json(
            {
                "price": price_res.model_dump(),
                "pe": pe_res.model_dump(),
                "analysis": analysis_res.model_dump(),
            }
        )

else:
    st.info("Enter a ticker symbol and click *Analyze* to begin.")
