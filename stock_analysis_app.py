#!/usr/bin/env python3
"""Small command‑line runner for the stock analysis agents.

This script preserves the original CLI behaviour while delegating the heavy
lifting to ``stock_analysis_lib``.  Keeping the CLI is useful for quick
testing and for automated jobs where a text output (JSON) is preferred over a
graphical UI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Load environment variables from a .env file that sits *alongside* this
# script so running it from any working directory still picks up the key.
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(dotenv_path=env_path, override=False)
except ImportError:  # pragma: no cover – optional dependency
    pass

from stock_analysis_lib import analyze_ticker


async def _main() -> None:  # noqa: D401
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Stock Financial Analysis (CLI)")
    parser.add_argument("ticker", help="Stock ticker symbol, e.g. AAPL")
    parser.add_argument("--model", default=None, help="Override LLM model")
    args = parser.parse_args()

    price_res, pe_res, analysis_res, judge_res = await analyze_ticker(args.ticker, model=args.model)

    # Dump everything as JSON for easy downstream piping / parsing.
    output = {
        "price": price_res.model_dump(),
        "pe": pe_res.model_dump(),
        "analysis": analysis_res.model_dump(),
        "judgement": judge_res.model_dump(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
