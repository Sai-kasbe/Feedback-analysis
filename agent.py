from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_ai_report(df, col):
    text_data = "\n".join(df[col].astype(str).head(2000))  
    # limit for performance

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
