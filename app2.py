import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- SETUP AND PATHS ---
CSV_DIR = r"C:/Users/ashwi/OneDrive/Guvi nb/project 2/csv_by_symbol"

# --- DATA LOADING FUNCTIONS ---
@st.cache_data
def load_stock_data(csv_dir):
    if not os.path.exists(csv_dir):
        st.error(f"Directory not found: {csv_dir}")
        return pd.DataFrame()
    
    dfs = []
    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(csv_dir, file))
            df["Date"] = pd.to_datetime(df["Date"])
            df["Symbol"] = file.replace(".csv", "")
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# --- ANALYTICS FUNCTIONS ---
def calculate_metrics(df):
    result = []
    for symbol, data in df.groupby("Symbol"):
        data = data.sort_values("Date")
        start = data.iloc[0]["Close"]
        end = data.iloc[-1]["Close"]
        yearly_return = ((end - start) / start) * 100
        result.append({
            "Symbol": symbol,
            "Return (%)": yearly_return,
            "Avg Price": data["Close"].mean(),
            "Avg Volume": data["Volume"].mean()
        })
    return pd.DataFrame(result).sort_values("Return (%)", ascending=False)

def calculate_volatility(df):
    vol = []
    for symbol, data in df.groupby("Symbol"):
        data = data.sort_values("Date")
        data["Daily Return"] = data["Close"].pct_change()
        vol.append({
            "Symbol": symbol,
            "Volatility": data["Daily Return"].std()
        })
    return pd.DataFrame(vol).sort_values("Volatility", ascending=False)

def calculate_cumulative_returns(df, top_symbols):
    subset = df[df["Symbol"].isin(top_symbols)].copy()
    subset = subset.sort_values(["Symbol", "Date"])
    subset["Daily Return"] = subset.groupby("Symbol")["Close"].pct_change()
    subset["Cumulative Return"] = subset.groupby("Symbol")["Daily Return"].transform(lambda x: (1 + x).cumprod())
    return subset

def calculate_monthly_performance(df):
    df['Month'] = df['Date'].dt.to_period('M').astype(str)
    monthly_data = []
    for (month, symbol), group in df.groupby(['Month', 'Symbol']):
        group = group.sort_values('Date')
        m_return = ((group.iloc[-1]['Close'] - group.iloc[0]['Close']) / group.iloc[0]['Close']) * 100
        monthly_data.append({'Month': month, 'Symbol': symbol, 'Return': m_return})
    return pd.DataFrame(monthly_data)

# --- STREAMLIT UI ---
st.set_page_config(page_title="Data driven stock analysis", layout="wide")
st.title("📈 Data driven stock analysis Dashboard")

df = load_stock_data(CSV_DIR)

if not df.empty:
    #  MARKET SUMMARY
    st.header("📊 Market Overview")
    
    metrics_df = calculate_metrics(df)
    total_stocks = metrics_df["Symbol"].nunique()
    green_stocks = len(metrics_df[metrics_df["Return (%)"] > 0])
    red_stocks = total_stocks - green_stocks
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Green Stocks", green_stocks)
    col2.metric("Red Stocks", red_stocks)
    col3.metric("Avg Market Price", f"₹{round(df['Close'].mean(), 2)}")
    col4.metric("Avg Market Volume", f"{int(df['Volume'].mean()):,}")

    # TOP GAINERS / LOSERS
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("🏆 Top 10 Gainers")
        st.dataframe(metrics_df.head(10), use_container_width=True)
    with col_right:
        st.subheader("🔻 Top 10 Losers")
        st.dataframe(metrics_df.tail(10), use_container_width=True)

    #  CUMULATIVE RETURNS 
    st.header("📈 Cumulative Returns (Top 5 Gainers)")
    top5_symbols = metrics_df.head(5)["Symbol"].tolist()
    cum_df = calculate_cumulative_returns(df, top5_symbols)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    for symbol in top5_symbols:
        symbol_data = cum_df[cum_df["Symbol"] == symbol]
        ax.plot(symbol_data["Date"], symbol_data["Cumulative Return"], label=symbol)
    ax.set_ylabel("Growth Factor (1.0 = Start)")
    ax.legend()
    st.pyplot(fig)

    #  VOLATILITY & CORRELATION 
    col_v, col_c = st.columns([1, 1])
    
    with col_v:
        st.header("📉 Top 10 Volatile Stocks")
        vol_df = calculate_volatility(df).head(10)
        fig2, ax2 = plt.subplots()
        sns.barplot(data=vol_df, x="Volatility", y="Symbol", ax=ax2, palette="magma")
        st.pyplot(fig2)

    with col_c:
        st.header("🔗 Correlation Heatmap")
        pivot = df.pivot(index="Date", columns="Symbol", values="Close")
        # Showing top 15 for readability
        top15 = metrics_df.head(15)["Symbol"]
        corr = pivot[top15].corr()
        fig3, ax3 = plt.subplots()
        sns.heatmap(corr, cmap="coolwarm", ax=ax3, annot=False)
        st.pyplot(fig3)

    # ---------------- MONTHLY PERFORMANCE ----------------
    st.header("📅 Monthly Top 5 Gainers & Losers")
    monthly_perf = calculate_monthly_performance(df)
    months = sorted(monthly_perf["Month"].unique(), reverse=True)
    
    selected_month = st.selectbox("Select Month", months)
    m_data = monthly_perf[monthly_perf["Month"] == selected_month]
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.write("**Top Monthly Gainers**")
        st.table(m_data.sort_values("Return", ascending=False).head(5))
    with m_col2:
        st.write("**Top Monthly Losers**")
        st.table(m_data.sort_values("Return").head(5))

else:
    st.warning("No data found. Please check your CSV path.")