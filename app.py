import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from agent import generate_ai_report
st.markdown("""
<style>

/* Background */
body {
    background-color: #f4f6f9;
}

/* Main title */
h1 {
    color: #1f2937;
    font-weight: 700;
}

/* Card style */
.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

/* Upload box */
[data-testid="stFileUploader"] {
    border: 2px dashed #6366f1;
    padding: 20px;
    border-radius: 12px;
    background-color: #eef2ff;
}

/* Metric styling */
.stMetric {
    background: white;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}

/* Section spacing */
.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)
# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Feedback Analyzer", layout="wide")

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.stMetric {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("""
<div style='padding:25px; border-radius:15px; background: linear-gradient(135deg, #6366f1, #8b5cf6); color:white'>
    <h1>🎯 AI Event Feedback Analyzer</h1>
    <p style='font-size:18px;'>Analyze thousands of feedbacks instantly with AI-powered insights</p>
</div>
""", unsafe_allow_html=True)

# ---------- FILE UPLOAD ----------
file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    # Select column
    feedback_col = st.selectbox("📝 Select Feedback Column", df.columns)

    # ---------- OPTIMIZATION ----------
    mode = st.radio("⚡ Processing Mode", ["Fast (Recommended)", "Full"])

    if mode == "Fast (Recommended)" and len(df) > 10000:
        st.warning("⚡ Large dataset detected — using optimized sampling (10,000 rows)")
        df = df.sample(10000)

    # ---------- ANALYSIS ----------
    analyzer = SentimentIntensityAnalyzer()

    texts = df[feedback_col].astype(str).tolist()
    scores = [analyzer.polarity_scores(t)['compound'] for t in texts]

    df['Score'] = scores
    df['Sentiment'] = df['Score'].apply(
        lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral")
    )

    # ---------- SUMMARY ----------
    st.markdown("## 📊 Event Overview")

    col1, col2, col3, col4 = st.columns(4)

    positive = (df['Sentiment']=="Positive").sum()
    negative = (df['Sentiment']=="Negative").sum()
    neutral = (df['Sentiment']=="Neutral").sum()
    total = len(df)

    score = round((positive/total)*100,2)

    col1.metric("😊 Positive", positive)
    col2.metric("😐 Neutral", neutral)
    col3.metric("😡 Negative", negative)
    col4.metric("🏆 Success Score", f"{score}%")

    # ---------- STATUS ----------
    if score > 70:
        st.success("🎉 Event was highly successful!")
    elif score > 50:
        st.warning("⚠️ Event was average, needs improvement")
    else:
        st.error("❌ Event needs major improvement")

    # ---------- CHARTS ----------
    st.markdown("## 📈 Visual Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        st.bar_chart(df['Sentiment'].value_counts())

    with col2:
        st.subheader("Sentiment Share")
        fig, ax = plt.subplots()
        df['Sentiment'].value_counts().plot.pie(
            autopct='%1.1f%%',
            colors=["#4CAF50", "#FFC107", "#F44336"],
            ax=ax
        )
        st.pyplot(fig)

    # ---------- TREND ----------
    st.markdown("## 📉 Trend Analysis")

    df['Smooth'] = df['Score'].rolling(50).mean()
    st.line_chart(df['Smooth'])

    # ---------- AI INSIGHTS ----------
    st.markdown("## 🤖 AI Recommendations")

    if st.button("🚀 Generate AI Insights"):
    st.info("⚡ Generating insights... please wait")

    report = generate_ai_report(df, feedback_col)

    st.success("Done")
    st.write(report)
