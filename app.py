import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------- PAGE ----------
st.set_page_config(page_title="AI Feedback Analyzer", layout="wide")

# ---------- HEADER ----------
st.markdown("""
<div style='padding:25px; border-radius:15px; background: linear-gradient(135deg, #6366f1, #8b5cf6); color:white'>
    <h1>🎯 AI Event Feedback Analyzer</h1>
    <p>Analyze feedback with real AI insights</p>
</div>
""", unsafe_allow_html=True)

# ---------- FILE ----------
file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    # ---------- COLUMN ----------
    feedback_col = st.selectbox("📝 Select Feedback Column", df.columns)

    # ---------- CLEAN ----------
    df[feedback_col] = df[feedback_col].astype(str).str.strip()
    df = df[df[feedback_col] != ""]
    df = df.dropna(subset=[feedback_col])
    df = df.reset_index(drop=True)

    # ---------- OPTIMIZE ----------
    if len(df) > 10000:
        st.warning("⚡ Large dataset detected — using sampling (10k rows)")
        df = df.sample(10000)

    # ---------- SENTIMENT ----------
    analyzer = SentimentIntensityAnalyzer()

    df["score"] = df[feedback_col].apply(
        lambda x: analyzer.polarity_scores(x)["compound"]
    )

    df["Sentiment"] = df["score"].apply(
        lambda x: "Positive" if x >= 0.05 else ("Negative" if x <= -0.05 else "Neutral")
    )

    # ---------- OVERVIEW ----------
    st.markdown("## 📊 Event Overview")

    pos = (df["Sentiment"]=="Positive").sum()
    neg = (df["Sentiment"]=="Negative").sum()
    neu = (df["Sentiment"]=="Neutral").sum()
    total = len(df)

    score = (pos / total) * 100

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("😊 Positive", pos)
    c2.metric("😐 Neutral", neu)
    c3.metric("😡 Negative", neg)
    c4.metric("🏆 Success Score", f"{score:.2f}%")

    # ---------- STATUS ----------
    if score > 70:
        st.success("🎉 Event was highly successful")
    elif score > 50:
        st.warning("⚠ Event average – needs improvement")
    else:
        st.error("❌ Event needs major improvement")

    # ---------- CHARTS ----------
    st.markdown("## 📊 Visual Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(df["Sentiment"].value_counts())

    with col2:
        fig, ax = plt.subplots()
        df["Sentiment"].value_counts().plot.pie(
            autopct="%1.1f%%",
            colors=["#4CAF50","#FFC107","#F44336"],
            ax=ax
        )
        st.pyplot(fig)

    # ---------- TREND ----------
    st.markdown("## 📈 Feedback Trend")

    batch_size = 200
    df["Batch"] = df.index // batch_size
    trend = df.groupby("Batch")["score"].mean()

    fig2, ax2 = plt.subplots()
    ax2.plot(trend, marker="o")
    ax2.set_title("Trend Over Time")
    st.pyplot(fig2)

    if len(trend) > 1:
        if trend.iloc[-1] > trend.iloc[0]:
            st.success("📈 Feedback Improving")
        else:
            st.error("📉 Feedback Declining")

    # ---------- AI INSIGHTS ----------
    st.markdown("## 🤖 AI Insights")

    # NEGATIVE
    neg_df = df[df["Sentiment"]=="Negative"]
    recommendations = []

    if len(neg_df) > 5:

        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=15)
        X = tfidf.fit_transform(neg_df[feedback_col])

        terms = tfidf.get_feature_names_out()
        scores = X.sum(axis=0).A1

        issues = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)

        st.write("### 🚨 Key Issues:")
        for issue, _ in issues[:5]:
            st.write(f"- {issue}")

            if "delay" in issue or "time" in issue:
                recommendations.append("Improve scheduling")

            elif "food" in issue:
                recommendations.append("Improve food quality")

            elif "management" in issue:
                recommendations.append("Improve organization")

    else:
        st.info("Not enough negative feedback")

    # POSITIVE
    pos_df = df[df["Sentiment"]=="Positive"]

    if len(pos_df) > 5:
        tfidf2 = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=10)
        X2 = tfidf2.fit_transform(pos_df[feedback_col])

        terms2 = tfidf2.get_feature_names_out()
        scores2 = X2.sum(axis=0).A1

        strengths = sorted(zip(terms2, scores2), key=lambda x: x[1], reverse=True)

        st.write("### 💚 Strengths:")
        for s, _ in strengths[:5]:
            st.write(f"- {s}")

    # ---------- FINAL ----------
    st.write("### 💡 Recommendations")

    if recommendations:
        for r in list(set(recommendations)):
            st.write(f"- {r}")
    else:
        st.write("- Maintain current performance")

    st.success("✅ Analysis Completed")
