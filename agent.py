import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_ai_report(df):
    text_data = "\n".join(df['feedback'].astype(str))

    prompt = f"""
    Analyze the following event feedback:

    {text_data}

    Provide:
    1. Overall sentiment summary
    2. Key problems
    3. Positive highlights
    4. Suggestions for improvement
    5. Ideas for future events
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response['choices'][0]['message']['content']
