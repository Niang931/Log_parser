from .models.gemini import GeminiParser
from .models.groq import GroqParser

MODELS = {
    "gemini": GeminiParser,
    "groq": GroqParser
}


# TODO: maybe turn this into an enum
def init_llm(provider="groq", **kwargs):
    return MODELS[provider](**kwargs)
