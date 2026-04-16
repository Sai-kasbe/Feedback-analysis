import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from agent import generate_ai_report

st.set_page_config(page_title="Feedback Analyzer", layout="wide")

st.title("📊 Event Feedback Analyzer (CSV Based)")

# Upload CSV
file = st.file_uploader("Upload Google Forms CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📄 Raw Data")
    st.dataframe(df)

    # Select feedback column
    feedback_col = st.selectbox("Select Feedback Column", df.columns)

    analyzer = SentimentIntensityAnalyzer()

    sentiments = []
    scores = []

    for text in df[feedback_col].astype(str):
        score = analyzer.polarity_scores(text)['compound']

        if score >= 0.05:
            sentiments.append("Positive")
        elif score <= -0.05:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")

        scores.append(score)

    df['Sentiment'] = sentiments
    df['Score'] = scores

    st.subheader("📊 Processed Data")
    st.dataframe(df)

    # ---------- VISUALS ----------

    col1, col2 = st.columns(2)

    # Bar Chart
    with col1:
        st.subheader("Sentiment Distribution")
        st.bar_chart(df['Sentiment'].value_counts())

    # Pie Chart
    with col2:
        st.subheader("Pie Chart")

        fig, ax = plt.subplots()
        df['Sentiment'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
        st.pyplot(fig)

    # Trend
    st.subheader("📈 Sentiment Trend")

    st.line_chart(df['Score'])

    # ---------- AI REPORT ----------
    st.subheader("🤖 AI Recommendations")

    if st.button("Generate AI Insights"):
        with st.spinner("Analyzing with AI..."):
            report = generate_ai_report(df)

        st.success("Analysis Complete")
        st.write(report)
