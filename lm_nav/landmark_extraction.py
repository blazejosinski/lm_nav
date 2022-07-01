import os
from typing import List
import openai
import spacy


PROMPT = """Take right next to an old white building. Look for a fire station, which you will see after passing by a school.
Ordered landmarks:
1. an old white building
2. a school
3. a fire station

Go straight for two blocks. Take right at a roundabout, before it you will pass a big, blue tree.
Ordered landmarks:
1. a big, blue tree
2. a roundabout

Look for a library, after taking a right turn next to a statue.
Ordered landmarks:
1. a statue
2. a library"""


def text_to_landmarks_spacy(text: str) -> List[str]:
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks if chunk.text.lower() != "you"]


def generic_language_model_api_call(
    api_base: str, api_key: str, model: str, text: str
) -> List[str]:
    openai.api_key = api_key
    openai.api_base = api_base

    prompt = PROMPT + "\n\n" + text + "\nOrdered landmarks:\n1."
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    landmark_text = response["choices"][0]["text"]
    landmarks = [s[s.find(". ") + 2 :] for s in landmark_text.split("\n")]
    return [landmark for landmark in landmarks if landmark]


def text_to_landmarks_gpt3(text: str) -> List[str]:
    return generic_language_model_api_call(
        "https://api.openai.com/v1",
        os.getenv("OPENAI_API_KEY"),
        "text-davinci-002",
        text,
    )


def text_to_landmarks_goose_ai(text: str, model: str) -> List[str]:
    return generic_language_model_api_call(
        "https://api.goose.ai/v1", os.getenv("GOOSE_API_KEY"), model, text
    )
