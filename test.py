
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    config={
        "system_instruction": """
        You are a helpful AI assistant.

        Formatting rules:
        1. Write answers in clear, complete paragraphs.
        2. Organize information logically.
        3. Use proper grammar and simple English.
        4. Avoid unnecessary bullet points and numbered lists.
        5. Use headings when the answer covers multiple topics.
        6. Keep paragraphs meaningful and well connected.
        7. Do not write fragmented sentences.
        8. Use bullet points only when listing items is genuinely useful.
        """
    },
    contents="what is the capital of nepal?"
)

print(response.text)
