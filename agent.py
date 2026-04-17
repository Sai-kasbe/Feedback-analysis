from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_ai_report(df, col):
    try:
        # Take small smart sample
        sample_df = df.sample(min(300, len(df)))

        text_data = "\n".join(sample_df[col].astype(str))

        prompt = f"""
        Analyze the following event feedback:

        {text_data}

        Provide:
        - Overall sentiment summary
        - Key problems
        - Positive highlights
        - Suggestions for improvement
        - Ideas for future events
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    except Exception as e:
        return "⚠️ AI service is busy or rate-limited. Please try again later."
