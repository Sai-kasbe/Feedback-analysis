from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- SMART SUMMARY FUNCTION ----------
def prepare_summary(df, col):
    positive = df[df['Sentiment']=="Positive"][col].head(20).tolist()
    negative = df[df['Sentiment']=="Negative"][col].head(20).tolist()
    neutral = df[df['Sentiment']=="Neutral"][col].head(10).tolist()

    summary = f"""
Positive Feedback Examples:
{positive}

Negative Feedback Examples:
{negative}

Neutral Feedback Examples:
{neutral}

Total Positive: {len(df[df['Sentiment']=="Positive"])}
Total Negative: {len(df[df['Sentiment']=="Negative"])}
Total Neutral: {len(df[df['Sentiment']=="Neutral"])}
"""
    return summary


# ---------- FINAL AI FUNCTION ----------
def generate_ai_report(df, col):
    try:
        summary = prepare_summary(df, col)

        prompt = f"""
You are an expert event analyst.

Analyze this summarized feedback data:

{summary}

Give output in this format:

1. Overall Performance:
2. Key Strengths:
3. Major Problems:
4. Suggestions for Improvement:
5. Final Verdict (Successful / Average / Needs Improvement)

Keep it clear, structured, and professional.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except Exception:
        return "⚠️ AI service busy or rate-limited. Please try again."
