"""
agent/llm.py
Turns raw tool output into an actual written answer, instead of returning
tool results verbatim. Two options are provided:

1. flan_t5_synthesize — local, no API key, reuses the same HuggingFace
   stack already used in rag_core.py. Lower quality but zero setup.
2. groq_synthesize — optional, much better quality, needs a free Groq API
   key (https://console.groq.com). Commented out below; uncomment and set
   GROQ_API_KEY as an environment variable to use it instead.

Pass either as Agent(synthesize_fn=...) in app.py.
"""
from functools import lru_cache

FLAN_MODEL_NAME = "google/flan-t5-base"
MAX_PROMPT_CHARS = 2500  # keep well within FLAN-T5's ~512 token context


@lru_cache(maxsize=1)
def _get_flan_pipeline():
    from transformers import pipeline
    return pipeline("text2text-generation", model=FLAN_MODEL_NAME)


def flan_t5_synthesize(prompt: str) -> str:
    """callable(prompt: str) -> str — pass to Agent(synthesize_fn=...)."""
    if len(prompt) > MAX_PROMPT_CHARS:
        # Keep the instruction + question, trim from the middle of the
        # findings block if needed, since that's usually the longest part.
        prompt = prompt[:MAX_PROMPT_CHARS]
    pipe = _get_flan_pipeline()
    result = pipe(prompt, max_new_tokens=200, do_sample=False)
    return result[0]["generated_text"].strip()


# ---------------------------------------------------------------------
# Optional: swap in Groq for noticeably better answers (still free tier,
# just requires an API key instead of zero setup). Uncomment to use.
# ---------------------------------------------------------------------
# import os
#
# def groq_synthesize(prompt: str) -> str:
#     from groq import Groq
#     client = Groq(api_key=os.environ["GROQ_API_KEY"])
#     response = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[{"role": "user", "content": prompt}],
#         max_tokens=300,
#     )
#     return response.choices[0].message.content.strip()