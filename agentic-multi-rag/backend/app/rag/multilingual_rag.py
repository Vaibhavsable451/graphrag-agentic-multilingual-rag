"""
Multilingual RAG: detects the query language, and (a) retrieves
using the shared multilingual embedding space directly when
possible, or (b) translates the query to English for retrieval
against English-heavy corpora and translates the answer back.
"""
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

DetectorFactory.seed = 0  # deterministic language detection


def detect_language(text: str) -> str:
    # langdetect is unreliable on very short strings (e.g. "hi" gets
    # misread as Swahili) — below this length, default to English
    # rather than trust a low-signal guess.
    if len(text.strip()) < 10:
        return "en"
    try:
        return detect(text)
    except Exception:  # noqa: BLE001
        return "en"


def translate(text: str, target_lang: str, source_lang: str = "auto") -> str:
    if target_lang == source_lang:
        return text
    try:
        return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
    except Exception:  # noqa: BLE001
        return text  # fail open — better a same-language answer than a crash


def prepare_multilingual_query(query: str) -> dict:
    """Returns the original query, its detected language, and an
    English version for retrieval fallback."""
    lang = detect_language(query)
    english_query = query if lang == "en" else translate(query, target_lang="en", source_lang=lang)
    return {"original": query, "lang": lang, "english": english_query}


def localize_answer(answer: str, target_lang: str) -> str:
    if target_lang == "en":
        return answer
    return translate(answer, target_lang=target_lang, source_lang="en")
