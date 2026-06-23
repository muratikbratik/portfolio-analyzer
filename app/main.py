import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from scipy import stats
from datetime import datetime, timedelta

st.set_page_config(page_title="Portfolio Risk Analyzer", page_icon="📈", layout="wide")

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0a0e1a; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0f1424; border-right: 1px solid #1e2d4a; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f1424 0%, #1a2744 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 28px;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .main-header p { color: #7a9cc0; margin: 0; font-size: 0.95rem; }
    .header-badge {
        display: inline-block;
        background: #1e3a5f;
        color: #4a9eff;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        border: 1px solid #2a5080;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #0f1424;
        border: 1px solid #1e2d4a;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="metric-container"] label { color: #7a9cc0 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.8px; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.6rem !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

    /* Section headers */
    .section-title {
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 8px 0 16px 0;
        padding-bottom: 10px;
        border-bottom: 1px solid #1e2d4a;
        letter-spacing: 0.2px;
    }

    /* AI Summary box */
    .ai-box {
        background: #0f1424;
        border: 1px solid #1e3a5f;
        border-left: 3px solid #4a9eff;
        border-radius: 10px;
        padding: 20px 24px;
        line-height: 1.9;
        color: #c8d8ea;
    }

    /* Divider */
    hr { border-color: #1e2d4a !important; }

    /* Sidebar labels */
    .stSidebar label { color: #7a9cc0 !important; font-size: 0.82rem !important; }
    .stSidebar .stSelectbox label, .stSidebar .stNumberInput label { color: #a0b8d0 !important; }
</style>

<div class="main-header">
    <p>
        <span class="header-badge">LIVE DATA</span>
        <span class="header-badge">MONTE CARLO</span>
        <span class="header-badge">AI ANALYSIS</span>
    </p>
    <h1>Portfolio Risk Analyzer</h1>
    <p>Real-time risk metrics, Monte Carlo simulation and AI-generated insights for your stock portfolio.</p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.header("Portfolio Settings")

default_tickers = "AAPL, MSFT, GOOGL, AMZN, TSLA"
tickers_input = st.sidebar.text_input("Stock Tickers (comma separated)", value=default_tickers)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

st.sidebar.markdown("**Portfolio Weights (%)**")
weights_raw = []
for ticker in tickers:
    w = st.sidebar.number_input(f"{ticker}", min_value=0, max_value=100, value=int(100 / len(tickers)), step=5)
    weights_raw.append(w)

total_weight = sum(weights_raw)
if total_weight == 0:
    st.error("Weights cannot all be zero.")
    st.stop()
weights = np.array(weights_raw) / total_weight

period = st.sidebar.selectbox("Historical Period", ["1y", "2y", "3y", "5y"], index=1)
investment = st.sidebar.number_input("Initial Investment ($)", value=10000, step=1000)
mc_simulations = 500

if st.sidebar.button("Analyze Portfolio", type="primary"):
    with st.spinner("Fetching market data..."):
        end = datetime.today()
        raw = yf.download(tickers + ["SPY"], period=period, auto_adjust=True, progress=False)["Close"]
        if raw.empty:
            st.error("Could not fetch data. Check your ticker symbols.")
            st.stop()
        raw = raw.dropna()

    spy = raw["SPY"]
    prices = raw[tickers]
    returns = prices.pct_change().dropna()
    spy_returns = spy.pct_change().dropna()

    port_returns = returns.dot(weights)
    cum_returns = (1 + port_returns).cumprod()
    spy_cum = (1 + spy_returns).cumprod()

    total_return = cum_returns.iloc[-1] - 1
    spy_total_return = spy_cum.iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(port_returns)) - 1
    annual_vol = port_returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    var_95 = np.percentile(port_returns, 5)
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    beta = np.cov(port_returns, spy_returns.loc[port_returns.index])[0][1] / np.var(spy_returns.loc[port_returns.index])

    # --- KPI ---
    st.markdown('<div class="section-title">KEY METRICS</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Return", f"{total_return:.1%}", f"vs S&P 500: {spy_total_return:.1%}")
    c2.metric("Annual Return", f"{annual_return:.1%}")
    c3.metric("Sharpe Ratio", f"{sharpe:.2f}", "Good > 1.0")
    c4.metric("VaR (95%)", f"{var_95:.2%}", "Daily worst case")
    c5.metric("Max Drawdown", f"{max_drawdown:.2%}")
    st.divider()

    # --- Cumulative Returns ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">PORTFOLIO vs S&P 500</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns * investment,
                                 name="Portfolio", line=dict(color="#4C78A8", width=2)))
        fig.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum * investment,
                                 name="S&P 500", line=dict(color="#F58518", width=2, dash="dot")))
        fig.update_layout(yaxis_title="Portfolio Value ($)", xaxis_title="", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">DRAWDOWN ANALYSIS</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100,
                                  fill="tozeroy", line=dict(color="#E45756"), name="Drawdown"))
        fig2.update_layout(yaxis_title="Drawdown (%)", xaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # --- Monte Carlo ---
    st.markdown('<div class="section-title">MONTE CARLO SIMULATION — 1 YEAR FORECAST (500 PATHS)</div>', unsafe_allow_html=True)
    mu = port_returns.mean()
    sigma = port_returns.std()
    days = 252
    mc_matrix = np.zeros((days, mc_simulations))
    for sim in range(mc_simulations):
        daily = np.random.normal(mu, sigma, days)
        mc_matrix[:, sim] = investment * np.cumprod(1 + daily)

    final_values = mc_matrix[-1, :]
    p10, p50, p90 = np.percentile(final_values, [10, 50, 90])

    fig3 = go.Figure()
    for i in range(min(100, mc_simulations)):
        fig3.add_trace(go.Scatter(y=mc_matrix[:, i], mode="lines",
                                  line=dict(width=0.5, color="rgba(76,120,168,0.15)"), showlegend=False))
    fig3.add_trace(go.Scatter(y=np.percentile(mc_matrix, 90, axis=1),
                              name="90th percentile", line=dict(color="green", width=2)))
    fig3.add_trace(go.Scatter(y=np.percentile(mc_matrix, 50, axis=1),
                              name="Median", line=dict(color="white", width=2)))
    fig3.add_trace(go.Scatter(y=np.percentile(mc_matrix, 10, axis=1),
                              name="10th percentile", line=dict(color="red", width=2)))
    fig3.update_layout(yaxis_title="Portfolio Value ($)", xaxis_title="Trading Days",
                       plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                       font=dict(color="white"), legend=dict(orientation="h"))
    st.plotly_chart(fig3, use_container_width=True)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Optimistic (90th %)", f"${p90:,.0f}", f"+{(p90/investment-1):.1%}")
    mc2.metric("Median outcome", f"${p50:,.0f}", f"+{(p50/investment-1):.1%}")
    mc3.metric("Pessimistic (10th %)", f"${p10:,.0f}", f"{(p10/investment-1):.1%}")
    st.divider()

    # --- Allocation ---
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-title">PORTFOLIO ALLOCATION</div>', unsafe_allow_html=True)
        fig4 = px.pie(names=tickers, values=weights * 100,
                      color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig4, use_container_width=True)

    with col4:
        st.markdown('<div class="section-title">RETURN DISTRIBUTION</div>', unsafe_allow_html=True)
        fig5 = go.Figure()
        fig5.add_trace(go.Histogram(x=port_returns * 100, nbinsx=50,
                                    marker_color="#4C78A8", name="Daily Returns"))
        fig5.add_vline(x=var_95 * 100, line_dash="dash", line_color="red",
                       annotation_text=f"VaR 95%: {var_95:.2%}")
        fig5.update_layout(xaxis_title="Daily Return (%)", yaxis_title="Frequency")
        st.plotly_chart(fig5, use_container_width=True)

    st.divider()

    # --- AI Summary ---
    st.markdown('<div class="section-title">AI PORTFOLIO ANALYSIS</div>', unsafe_allow_html=True)

    def generate_summary(sharpe, total_return, spy_total_return, max_drawdown, var_95, beta, annual_vol, p50, investment):
        lines = []
        outperform = total_return - spy_total_return
        if outperform > 0:
            lines.append(f"✅ Your portfolio **outperformed the S&P 500 by {outperform:.1%}** over the selected period.")
        else:
            lines.append(f"⚠️ Your portfolio **underperformed the S&P 500 by {abs(outperform):.1%}** over the selected period.")

        if sharpe >= 1.5:
            lines.append(f"✅ **Sharpe Ratio of {sharpe:.2f}** is excellent — strong risk-adjusted returns.")
        elif sharpe >= 1.0:
            lines.append(f"✅ **Sharpe Ratio of {sharpe:.2f}** is good — acceptable risk-adjusted performance.")
        elif sharpe >= 0.5:
            lines.append(f"⚠️ **Sharpe Ratio of {sharpe:.2f}** is moderate — returns may not justify the risk taken.")
        else:
            lines.append(f"🔴 **Sharpe Ratio of {sharpe:.2f}** is poor — consider rebalancing your portfolio.")

        if abs(max_drawdown) > 0.30:
            lines.append(f"🔴 **Max Drawdown of {max_drawdown:.1%}** is high — portfolio experienced severe losses at peak.")
        elif abs(max_drawdown) > 0.15:
            lines.append(f"⚠️ **Max Drawdown of {max_drawdown:.1%}** — moderate historical loss from peak.")
        else:
            lines.append(f"✅ **Max Drawdown of {max_drawdown:.1%}** is well-controlled.")

        lines.append(f"📊 **Daily VaR (95%):** On a typical bad day, expect a loss of up to **{abs(var_95):.2%}** — that's **${abs(var_95) * investment:,.0f}** on a ${investment:,} investment.")

        if beta > 1.3:
            lines.append(f"⚠️ **Beta of {beta:.2f}** — portfolio is significantly more volatile than the market.")
        elif beta > 0.8:
            lines.append(f"✅ **Beta of {beta:.2f}** — portfolio moves broadly in line with the market.")
        else:
            lines.append(f"✅ **Beta of {beta:.2f}** — portfolio is less volatile than the market (defensive).")

        lines.append(f"🎲 **Monte Carlo forecast:** In the median scenario, your ${investment:,} grows to **${p50:,.0f}** over the next 12 months.")
        return "\n\n".join(lines)

    summary = generate_summary(sharpe, total_return, spy_total_return, max_drawdown, var_95, beta, annual_vol, p50, investment)
    st.markdown(f'<div class="ai-box">{summary}</div>', unsafe_allow_html=True)
