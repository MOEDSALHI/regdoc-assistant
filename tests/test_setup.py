import os

from dotenv import load_dotenv

# from mistralai import Mistral
from mistralai.client import Mistral

load_dotenv()


def test_mistral_connection():
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    response = client.chat.complete(
        model="mistral-small-latest", messages=[{"role": "user", "content": "Réponds juste OK"}]
    )
    print(response.choices[0].message.content)
    assert response.choices[0].message.content is not None
