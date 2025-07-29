from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def generate_reflection(tafsir_text: str, lang: str = "en") -> str:
    client = OpenAI()

    prompt = (
        f"Based on the following Arabic Islamic tafsir text, write a spiritual reflection "
        f"in {lang} that helps the reader draw practical wisdom and guidance. Focus on character, morality, or life purpose.\n\n"
        f"{tafsir_text}"
    )

    response = client.responses.create(
        model="gpt-4o",
        input=prompt,
        temperature=0.6
    )
    return response.output_text.strip()