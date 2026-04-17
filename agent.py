from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_ai_report(df, col):
    try:
        # Small sample to avoid limit
        sample = df.sample(min(200, len(df)))

        text_data = "\n".join(sample[col].astype(str))

        prompt = f"""
        Analyze this feedback and give:

        - Overall performance
        - Key strengths
        - Major problems
        - Suggestions for improvement
        - Final verdict
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except Exception:
        # 🔥 FALLBACK SYSTEM (VERY IMPORTANT)
        positive = (df['Sentiment']=="Positive").sum()
        negative = (df['Sentiment']=="Negative").sum()
        neutral = (df['Sentiment']=="Neutral").sum()

        total = len(df)
        score = (positive/total)*100

        result = f"""
AI Summary (Fallback Mode):

Overall Performance:
Success Rate: {round(score,2)}%

Key Strengths:
- High positive feedback from users

Major Problems:
- Negative feedback indicates issues in some areas

Suggestions:
- Improve weak areas identified
- Maintain strong aspects of event

Final Verdict:
{"Successful" if score > 70 else "Average" if score > 50 else "Needs Improvement"}
"""
        return result
