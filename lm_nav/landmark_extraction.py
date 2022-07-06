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

SIMPLIFIED_PROMPT = """Look for a library, after taking a right turn next to a statue.
Landmarks:
1. a statue
2. a library

Look for a statue. Then look for a library. Then go towards a pink house.
Landmarks:
1. a statue
2. a library
3. a pink house"""


def remove_article(string):
    article = ["a", "an", "the"]
    words = string.split()
    none_articles = [w for w in words if w not in article]
    return " ".join(none_articles)


def text_to_landmarks_spacy(text: str) -> List[str]:
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(text)
    black_list = ["you", "left", "right", "left turn", "right turn"]
    return [chunk.text for chunk in doc.noun_chunks if remove_article(chunk.text.lower()) not in black_list]


def generic_language_model_api_call(
    api_base: str, api_key: str, model: str, text: str, postprocess: bool = False, simple_prompt: bool = False
) -> List[str]:
    openai.api_key = api_key
    openai.api_base = api_base

    prompt = SIMPLIFIED_PROMPT if simple_prompt else PROMPT
    prompt += "\n\n" + text + "\n"
    prompt += "Landmarks" if simple_prompt else "Ordered landmarks"
    prompt += ":\n1."
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    landmark_text = response["choices"][0]["text"]
    landmarks = [s[s.find(". ") + 2 :] for s in landmark_text.split("\n")]
    if postprocess:
        for i in range(len(landmarks)):
            if len(landmarks[i]) == 0:
                landmarks = landmarks[:i]
                break
    return [landmark for landmark in landmarks if landmark]


def text_to_landmarks_gpt3(text: str, simple_prompt: bool = False) -> List[str]:
    return generic_language_model_api_call(
        "https://api.openai.com/v1",
        os.getenv("OPENAI_API_KEY"),
        "text-davinci-002",
        text,
        simple_prompt = simple_prompt,
    )


def text_to_landmarks_goose_ai(text: str, model: str, simple_prompt: bool = False) -> List[str]:
    return generic_language_model_api_call(
        "https://api.goose.ai/v1", os.getenv("GOOSE_API_KEY"), model, text, postprocess=True,
        simple_prompt = simple_prompt,
    )
