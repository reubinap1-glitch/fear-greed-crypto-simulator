# app.py
# Fear & Greed Crypto Simulator
# ---------------------------------------------------------
# Run locally:
#   pip install -r requirements.txt
#   streamlit run app.py
#
# Optional:
#   Add your CoinMarketCap API key in Streamlit secrets:
#   .streamlit/secrets.toml
#
#   CMC_API_KEY="YOUR_KEY"
#
# ---------------------------------------------------------

import os
import io
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fear & Greed Crypto Simulator",
    page_icon="📈",
    layout="wide",
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 1.5rem;
}

.metric-card {
    border-radius: 16px;
    padding: 1rem;
    background: #151922;
    border: 1px solid rgba(255,255,255,0.08);
}

.disclaimer {
    border-left: 4px solid #ff9800;
    padding: 1rem;
    background: rgba(255,152,0,0.08);
    border-radius: 8px;
    font-size: 0.9rem;
}

.small-note {
    font-size: 0.85rem;
    opacity: 0.75;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# CONSTANTS
# =========================================================

START_DATE = "2018-02-01"

ZONE_COLORS = {
    "Extreme Fear": "#8b0000",
    "Fear": "#ff6b6b",
    "Neutral": "#f4d35e",
    "Greed": "#4caf50",
    "Extreme Greed": "#006400",
}

# =========================================================
# HELPERS
# =========================================================

def classify_index(value):
    value = float(value)

    if value <= 24:
        return "Extreme Fear"
    elif value <= 44:
        return "Fear"
    elif value <= 55:
        return "Neutral"
    elif value <= 74:
        return "Greed"
    else:
        return "Extreme Greed"


def zone_color(value):
    return ZONE_COLORS[classify_index(value)]


def compute_max_drawdown(equity_curve):
    roll_max = equity_curve.cummax()
    drawdown = equity_curve / roll_max - 1.0
    return drawdown.min() * 100


def gauge_chart(title, value):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": zone_color(value)},
                "steps": [
                    {"range": [0, 25], "color": "#8b0000"},
                    {"range": [25, 45], "color": "#ff6b6b"},
                    {"range": [45, 55], "color": "#f4d35e"},
                    {"range": [55, 75], "color": "#4caf50"},
                    {"range": [75, 100], "color": "#006400"},
                ],
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="#151922",
        plot_bgcolor="#151922",
        font=dict(color="white"),
        height=320,
        margin=dict(t=60, b=20, l=20, r=20),
    )

    return fig


# =========================================================
# DATA FETCHING
# =========================================================

@st.cache_data(ttl=3600)
def fetch_alternative_me_current():
    url = "https://api.alternative.me/fng/?limit=1"
    r = requests.get(url, timeout=20)
    data = r.json()["data"][0]

    return {
        "value": int(data["value"]),
        "classification": data["value_classification"],
        "timestamp": pd.to_datetime(int(data["timestamp"]), unit="s")
    }


@st.cache_data(ttl=3600)
def fetch_alternative_me_historical():
    url = "https://api.alternative.me/fng/?limit=0"
    r = requests.get(url, timeout=20)

    data = r.json()["data"]

    df = pd.DataFrame(data)

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    df["value"] = df["value"].astype(float)

    df = df.rename(columns={
        "timestamp": "date",
        "value": "alt_fng"
    })

    df = df[["date", "alt_fng"]]
    df = df.sort_values("date")
    df = df[df["date"] >= START_DATE]

    return df


@st.cache_data(ttl=3600)
def fetch_btc_price():
    """
    Uses CoinGecko public market chart endpoint
    """
    url = (
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        "?vs_currency=usd&days=max&interval=daily"
    )

    r = requests.get(url, timeout=30)
    prices = r.json()["prices"]

    df = pd.DataFrame(prices, columns=["timestamp", "btc_price"])

    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
    df["date"] = pd.to_datetime(df["date"])

    df = df.groupby("date", as_index=False)["btc_price"].mean()
    df = df[df["date"] >= START_DATE]

    return df


@st.cache_data(ttl=3600)
def fetch_cmc_fng_current(api_key):
    """
    CoinMarketCap Fear & Greed endpoint.
    Requires free API key.
    """

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": api_key,
    }

    url = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/latest"

    r = requests.get(url, headers=headers, timeout=20)

    if r.status_code != 200:
        return None

    data = r.json()["data"]

    return {
        "value": float(data["value"]),
        "classification": data["value_classification"],
    }


@st.cache_data(ttl=3600)
def fetch_cmc_fng_historical(api_key):

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": api_key,
    }

    url = "https://pro-api.coinmarketcap.com/v3/fear-and-greed/historical?limit=5000"

    r = requests.get(url, headers=headers, timeout=30)

    if r.status_code != 200:
        return None

    raw = r.json()["data"]

    rows = []

    for item in raw:
        rows.append({
            "date": pd.to_datetime(item["timestamp"]).normalize(),
            "cmc_fng": float(item["value"]),
        })

    df = pd.DataFrame(rows)

    df = df.sort_values("date")
    df = df[df["date"] >= START_DATE]

    return df


# =========================================================
# STRATEGY BACKTEST
# =========================================================

def run_backtest(
    df,
    buy_threshold,
    sell_threshold,
    daily_buy_amount,
    daily_sell_pct,
    selected_index
):

    btc = 0.0
    cash_invested = 0.0
    cash_realized = 0.0

    trades = []

    equity_curve = []

    for _, row in df.iterrows():

        price = row["btc_price"]
        sentiment = row[selected_index]

        if pd.isna(sentiment):
            continue

        # BUY LOGIC
        if sentiment <= buy_threshold:
            btc_bought = daily_buy_amount / price
            btc += btc_bought
            cash_invested += daily_buy_amount

            trades.append({
                "date": row["date"],
                "type": "BUY",
                "price": price,
                "sentiment": sentiment,
                "btc": btc_bought,
            })

        # SELL LOGIC
        elif sentiment >= sell_threshold and btc > 0:

            btc_to_sell = btc * daily_sell_pct
            usd_received = btc_to_sell * price

            btc -= btc_to_sell
            cash_realized += usd_received

            trades.append({
                "date": row["date"],
                "type": "SELL",
                "price": price,
                "sentiment": sentiment,
                "btc": btc_to_sell,
            })

        portfolio_value = cash_realized + btc * price

        equity_curve.append({
            "date": row["date"],
            "equity": portfolio_value
        })

    final_value = cash_realized + btc * df.iloc[-1]["btc_price"]

    total_return = (
        (final_value - cash_invested) / cash_invested * 100
        if cash_invested > 0 else 0
    )

    equity_df = pd.DataFrame(equity_curve)

    max_dd = compute_max_drawdown(equity_df["equity"])

    sells = [t for t in trades if t["type"] == "SELL"]

    profitable_sells = 0

    for s in sells:
        if s["price"] > df["btc_price"].mean():
            profitable_sells += 1

    win_rate = (
        profitable_sells / len(sells) * 100
        if len(sells) > 0 else 0
    )

    return {
        "final_value": final_value,
        "cash_invested": cash_invested,
        "btc_remaining": btc,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "trades": pd.DataFrame(trades),
        "equity_curve": equity_df,
    }


# =========================================================
# TITLE
# =========================================================

st.title("📈 Fear & Greed Crypto Simulator")

st.markdown("""
Interactive crypto sentiment analytics + educational backtesting laboratory.

Explore how Bitcoin price action correlates with market psychology using:
- Alternative.me Crypto Fear & Greed Index
- CoinMarketCap Fear & Greed Index
- BTC historical price overlays
- Strategy simulation engine
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Configuration")

cmc_api_key = st.sidebar.text_input(
    "CoinMarketCap API Key (Optional)",
    type="password",
    value=st.secrets.get("CMC_API_KEY", "")
)

selected_index = st.sidebar.selectbox(
    "Primary Strategy Index",
    ["alt_fng", "cmc_fng"]
)

buy_threshold = st.sidebar.slider(
    "Buy Threshold (Fear)",
    0,
    50,
    25
)

sell_threshold = st.sidebar.slider(
    "Sell Threshold (Greed)",
    50,
    100,
    75
)

daily_buy_amount = st.sidebar.number_input(
    "Daily DCA Buy Amount ($)",
    value=25.0,
    min_value=1.0
)

daily_sell_pct = st.sidebar.slider(
    "Daily Sell Percentage",
    0.01,
    1.0,
    0.10
)

# =========================================================
# LOAD DATA
# =========================================================

with st.spinner("Loading market + sentiment data..."):

    alt_current = fetch_alternative_me_current()
    alt_hist = fetch_alternative_me_historical()
    btc_df = fetch_btc_price()

    cmc_current = None
    cmc_hist = None

    if cmc_api_key:
        try:
            cmc_current = fetch_cmc_fng_current(cmc_api_key)
            cmc_hist = fetch_cmc_fng_historical(cmc_api_key)
        except Exception:
            st.warning("Unable to load CoinMarketCap data.")

# =========================================================
# CURRENT GAUGES
# =========================================================

st.subheader("🧠 Current Market Sentiment")

col1, col2 = st.columns(2)

with col1:

    st.plotly_chart(
        gauge_chart(
            "Alternative.me Fear & Greed",
            alt_current["value"]
        ),
        use_container_width=True
    )

    st.success(
        f"{alt_current['classification']} ({alt_current['value']})"
    )

with col2:

    if cmc_current:

        st.plotly_chart(
            gauge_chart(
                "CoinMarketCap Fear & Greed",
                cmc_current["value"]
            ),
            use_container_width=True
        )

        st.success(
            f"{cmc_current['classification']} ({cmc_current['value']})"
        )

    else:
        st.info(
            "Add a CoinMarketCap API key to enable CMC Fear & Greed data."
        )

# =========================================================
# MERGE DATA
# =========================================================

df = btc_df.merge(alt_hist, on="date", how="left")

if cmc_hist is not None:
    df = df.merge(cmc_hist, on="date", how="left")
else:
    df["cmc_fng"] = np.nan

# =========================================================
# HISTORICAL CHART
# =========================================================

st.subheader("📊 BTC Price vs Fear & Greed History")

fig = make_subplots(specs=[[{"secondary_y": True}]])

# BTC
fig.add_trace(
    go.Scatter(
        x=df["date"],
        y=df["btc_price"],
        name="BTC Price",
        line=dict(width=2),
    ),
    secondary_y=False
)

# Alternative.me
fig.add_trace(
    go.Scatter(
        x=df["date"],
        y=df["alt_fng"],
        name="Alternative.me F&G",
        line=dict(dash="dot"),
    ),
    secondary_y=True
)

# CMC
if cmc_hist is not None:
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["cmc_fng"],
            name="CMC F&G",
            line=dict(dash="dash"),
        ),
        secondary_y=True
    )

# Sentiment zones
zones = [
    (0, 25, "#8b0000"),
    (25, 45, "#ff6b6b"),
    (45, 55, "#f4d35e"),
    (55, 75, "#4caf50"),
    (75, 100, "#006400"),
]

for low, high, color in zones:
    fig.add_hrect(
        y0=low,
        y1=high,
        fillcolor=color,
        opacity=0.08,
        line_width=0,
        secondary_y=True
    )

fig.update_layout(
    height=700,
    template="plotly_dark",
    hovermode="x unified",
    legend_orientation="h",
    title="Bitcoin Price vs Market Sentiment"
)

fig.update_yaxes(
    title_text="BTC Price (USD)",
    secondary_y=False,
    type="log"
)

fig.update_yaxes(
    title_text="Fear & Greed Index",
    secondary_y=True,
    range=[0, 100]
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# BACKTEST
# =========================================================

st.subheader("🧪 Strategy Backtester")

results = run_backtest(
    df=df,
    buy_threshold=buy_threshold,
    sell_threshold=sell_threshold,
    daily_buy_amount=daily_buy_amount,
    daily_sell_pct=daily_sell_pct,
    selected_index=selected_index
)

# =========================================================
# METRICS
# =========================================================

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Total Return",
    f"{results['total_return']:.2f}%"
)

m2.metric(
    "Final Portfolio",
    f"${results['final_value']:,.0f}"
)

m3.metric(
    "Max Drawdown",
    f"{results['max_drawdown']:.2f}%"
)

m4.metric(
    "Win Rate",
    f"{results['win_rate']:.2f}%"
)

# =========================================================
# EQUITY CURVE
# =========================================================

equity_fig = go.Figure()

equity_fig.add_trace(
    go.Scatter(
        x=results["equity_curve"]["date"],
        y=results["equity_curve"]["equity"],
        name="Portfolio Value"
    )
)

equity_fig.update_layout(
    title="Strategy Equity Curve",
    template="plotly_dark",
    height=500
)

st.plotly_chart(equity_fig, use_container_width=True)

# =========================================================
# TRADE LOG
# =========================================================

st.subheader("📒 Trade Log")

trade_df = results["trades"]

if not trade_df.empty:
    st.dataframe(trade_df, use_container_width=True)
else:
    st.info("No trades generated with current settings.")

# =========================================================
# EXPORTS
# =========================================================

st.subheader("⬇️ Export Data")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Historical Dataset CSV",
    csv,
    file_name="fear_greed_crypto_data.csv",
    mime="text/csv"
)

if not trade_df.empty:

    trade_csv = trade_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Trade Log CSV",
        trade_csv,
        file_name="fear_greed_trade_log.csv",
        mime="text/csv"
    )

# =========================================================
# EDUCATIONAL SECTION
# =========================================================

with st.expander("📚 Educational Notes"):

    st.markdown("""
### Fear & Greed Interpretation

| Zone | Interpretation |
|---|---|
| 0-24 | Extreme Fear |
| 25-44 | Fear |
| 45-55 | Neutral |
| 56-74 | Greed |
| 75-100 | Extreme Greed |

### Strategy Logic

This simulator demonstrates a simplified sentiment-driven strategy:

- Buy BTC daily during fear conditions
- Sell portions during greed conditions
- Track portfolio performance over time

This is intended for:
- education
- experimentation
- behavioral finance research

—not financial advice.

### Important Caveats

- Historical performance does NOT guarantee future results
- Fear & Greed metrics are lagging sentiment indicators
- Real trading involves slippage, taxes, spreads, and fees
- Simplified backtests may overestimate strategy robustness
""")

# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("""
<div class="disclaimer">
<b>Disclaimer:</b><br>
This dashboard is provided strictly for educational and informational purposes.
It does not constitute investment advice, portfolio management, financial
planning, or trading recommendations. Cryptocurrency markets are highly volatile
and risky.
</div>
""", unsafe_allow_html=True)
