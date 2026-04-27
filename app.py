import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------- PAGE ----------
st.set_page_config(page_title="Feedback Analyzer", layout="wide")

# ---------- TITLE ----------
st.title("🎯 Simple AI Feedback Analyzer (Demo)")

# ---------- FILE ----------
file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # ---------- SELECT COLUMN ----------
    col = st.selectbox("Select Feedback Column", df.columns)

    # ---------- CLEAN ----------
    df[col] = df[col].astype(str)

    # ---------- SENTIMENT ----------
    analyzer = SentimentIntensityAnalyzer()

    df["score"] = df[col].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )

    df["Sentiment"] = df["score"].apply(
        lambda x: "Positive" if x >= 0.05 else (
            "Negative" if x <= -0.05 else "Neutral"
        )
    )

    # ---------- METRICS ----------
    st.subheader("📊 Overview")

    pos = (df["Sentiment"]=="Positive").sum()
    neg = (df["Sentiment"]=="Negative").sum()
    neu = (df["Sentiment"]=="Neutral").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("😊 Positive", pos)
    c2.metric("😐 Neutral", neu)
    c3.metric("😡 Negative", neg)

    # ---------- CHART ----------
    st.subheader("📊 Sentiment Distribution")
    st.bar_chart(df["Sentiment"].value_counts())

    # ---------- SIMPLE TREND ----------
    st.subheader("📈 Simple Trend")

    df = df.reset_index(drop=True)
    df["Batch"] = df.index // 100

    trend = df.groupby("Batch")["score"].mean()

    st.line_chart(trend)

    # ---------- SIMPLE INSIGHTS ----------
    st.subheader("💡 Basic Insights")

    total = len(df)
    success = (pos / total) * 100

    st.write(f"Success Score: {success:.2f}%")

    if success > 70:
        st.success("Event is Good 👍")
    elif success > 50:
        st.warning("Event is Average ⚠️")
    else:
        st.error("Event needs improvement ❌")
