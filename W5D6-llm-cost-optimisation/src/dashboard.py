import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.config import COST_LOG_FILE

COLUMNS = [
    "timestamp",
    "query",
    "model",
    "input_tokens",
    "output_tokens",
    "cache_hit",
    "compression_used",
    "cost_usd",
    "cost_inr",
]


def log_request(
    query: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_hit: bool,
    compression_used: bool,
    cost_usd: float,
    cost_inr: float,
    filename: str = COST_LOG_FILE,
) -> None:
    """
    Append one LLM request to the cost log CSV.
    """

    row = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_hit": cache_hit,
        "compression_used": compression_used,
        "cost_usd": cost_usd,
        "cost_inr": cost_inr,
    }

    df = pd.DataFrame(
        [row],
        columns=COLUMNS,
    )

    file_exists = os.path.exists(filename)

    df.to_csv(
        filename,
        mode="a",
        header=not file_exists,
        index=False,
    )


def load_cost_log(
    filename: str = COST_LOG_FILE,
) -> pd.DataFrame:
    """
    Load the LLM cost log.
    """

    if not os.path.exists(filename):
        return pd.DataFrame(columns=COLUMNS)

    return pd.read_csv(filename)


def calculate_summary(
    filename: str = COST_LOG_FILE,
) -> dict:
    """
    Calculate summary statistics for the dashboard.
    """

    df = load_cost_log(filename)

    if df.empty:
        return {
            "total_requests": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "cache_hits": 0,
            "cache_hit_rate": 0.0,
            "total_cost_usd": 0.0,
            "total_cost_inr": 0.0,
        }

    total_requests = len(df)

    cache_hits = int(
        df["cache_hit"]
        .astype(str)
        .str.lower()
        .eq("true")
        .sum()
    )

    total_input_tokens = int(
        df["input_tokens"].sum()
    )

    total_output_tokens = int(
        df["output_tokens"].sum()
    )

    total_cost_usd = float(
        df["cost_usd"].sum()
    )

    total_cost_inr = float(
        df["cost_inr"].sum()
    )

    cache_hit_rate = (
        cache_hits / total_requests * 100
        if total_requests > 0
        else 0.0
    )

    return {
        "total_requests": total_requests,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "cache_hits": cache_hits,
        "cache_hit_rate": cache_hit_rate,
        "total_cost_usd": total_cost_usd,
        "total_cost_inr": total_cost_inr,
    }


def run_dashboard():
    """
    Start the Streamlit dashboard.
    """

    import streamlit as st

    st.set_page_config(
        page_title="LLM Cost Optimisation",
        page_icon="💰",
        layout="wide",
    )

    st.title(
        "💰 LLM Cost Optimisation Dashboard"
    )

    st.caption(
        "W5D6 — Token Economics, Semantic Caching "
        "and Cost Tracking"
    )

    df = load_cost_log()

    # --------------------------------------------------
    # No data yet
    # --------------------------------------------------

    if df.empty:

        st.info(
            "No request data available yet."
        )

        st.markdown(
            """
            ### Dashboard is ready ✅

            The dashboard will display:

            - Total requests
            - Input tokens
            - Output tokens
            - Cache hit rate
            - USD cost
            - INR cost
            - Daily cost chart
            - Request history

            Run the W5D6 token audit or LLM pipeline
            to generate cost records.

            Cost records are stored in:

            `llm_cost_log.csv`
            """
        )

        return

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    summary = calculate_summary()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Requests",
            summary["total_requests"],
        )

    with col2:
        st.metric(
            "Cache Hit Rate",
            f"{summary['cache_hit_rate']:.2f}%",
        )

    with col3:
        total_tokens = (
            summary["total_input_tokens"]
            + summary["total_output_tokens"]
        )

        st.metric(
            "Total Tokens",
            f"{total_tokens:,}",
        )

    with col4:
        st.metric(
            "Total Cost",
            f"₹{summary['total_cost_inr']:.4f}",
        )

    st.divider()

    # --------------------------------------------------
    # Token statistics
    # --------------------------------------------------

    st.subheader("📊 Token Usage")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Input Tokens",
            f"{summary['total_input_tokens']:,}",
        )

    with col2:
        st.metric(
            "Output Tokens",
            f"{summary['total_output_tokens']:,}",
        )

    st.divider()

    # --------------------------------------------------
    # Daily cost
    # --------------------------------------------------

    st.subheader("📈 Daily Cost")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    daily_cost = (
        df.groupby(
            df["timestamp"].dt.date
        )["cost_inr"]
        .sum()
    )

    st.line_chart(daily_cost)

    st.divider()

    # --------------------------------------------------
    # Model cost
    # --------------------------------------------------

    st.subheader("🤖 Cost by Model")

    model_cost = (
        df.groupby("model")["cost_inr"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(model_cost)

    st.divider()

    # --------------------------------------------------
    # Request log
    # --------------------------------------------------

    st.subheader("📋 Request Log")

    st.dataframe(
        df,
        use_container_width=True,
    )


if __name__ == "__main__":
    run_dashboard()