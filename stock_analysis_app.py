#!/usr/bin/env python3
"""
Stock Financial Analysis App using OpenAI Agents SDK
"""
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
import argparse
import asyncio

from pydantic import BaseModel
from agents import Agent, Runner, WebSearchTool
from agents.tool import function_tool
from agents.model_settings import ModelSettings


class PriceResult(BaseModel):
    ticker: str
    price: float

# Agent to fetch stock price via web search tool (OpenAI Responses API)
PRICE_PROMPT = (
    "You are an assistant that fetches the latest stock price for a given ticker symbol. "
    "Use the provided web search tool to look up the current price and return JSON: {\"ticker\": TICKER, \"price\": PRICE}."
)
price_agent = Agent(
    name="PriceAgent",
    instructions=PRICE_PROMPT,
    tools=[WebSearchTool()],
    model_settings=ModelSettings(tool_choice="required"),
    output_type=PriceResult,
)

class PERatiosResult(BaseModel):
    ticker: str
    trailingPE: float | None = None
    forwardPE: float | None = None

@function_tool
def get_pe_ratios(ticker: str) -> PERatiosResult:
    """Fetch trailing and forward P/E ratios using yfinance (free, no API key required)."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    info = t.info or {}
    return PERatiosResult(
        ticker=ticker,
        trailingPE=info.get("trailingPE"),
        forwardPE=info.get("forwardPE"),
    )

class AnalysisResult(BaseModel):
    ticker: str
    price: float
    trailingPE: float | None
    forwardPE: float | None
    analysis: str

async def main():
    # Ensure API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable.")
        exit(1)

    parser = argparse.ArgumentParser(description="Stock Financial Analysis App")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("--model", default=None, help="LLM model to use (default by agent)")
    args = parser.parse_args()

    # Fetch latest price
    price_result = await Runner.run(price_agent, args.ticker)
    price_data = price_result.final_output

    # Fetch P/E ratios
    pe_result = await Runner.run(
        Agent(
            name="PERatioAgent",
            instructions="You are an assistant that returns P/E ratios for a given ticker.",
            tools=[get_pe_ratios],
            model_settings=ModelSettings(tool_choice="required"),
            output_type=PERatiosResult,
        ),
        args.ticker,
    )
    pe_data = pe_result.final_output

    # Analysis agent using both price and P/E data
    ANALYSIS_PROMPT = (
        "You are a financial analyst. Given the following data, provide a detailed analysis of "
        "the company's outlook:\n"
        "Ticker: {ticker}\n"
        "Latest Price: {price}\n"
        "Trailing P/E: {trailingPE}\n"
        "Forward P/E: {forwardPE}\n"
        "Return JSON matching the schema: {{\"ticker\": str, \"price\": float, "
        "\"trailingPE\": float|null, \"forwardPE\": float|null, \"analysis\": str}}."
    )
    analysis_instructions = ANALYSIS_PROMPT.format(
        ticker=args.ticker,
        price=price_data.price,
        trailingPE=pe_data.trailingPE,
        forwardPE=pe_data.forwardPE,
    )
    analysis_agent = Agent(
        name="AnalysisAgent",
        instructions=analysis_instructions,
        tools=[],
        output_type=AnalysisResult,
        model=args.model,
    )
    analysis_run = await Runner.run(analysis_agent, "")
    analysis = analysis_run.final_output
    # Use Pydantic V2 API for JSON serialization
    print(analysis.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())