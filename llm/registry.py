from .models.gemini import GeminiParser
from .models.groq import GroqParser

MODELS = {
    "gemini": GeminiParser,
    "groq": GroqParser
}


def init_llm(provider="groq", **kwargs):
    return MODELS[provider](**kwargs)
