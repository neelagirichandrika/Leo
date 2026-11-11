
import re
import spacy

# Load the English NLP model once when the application starts.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None


# Messages that normally need a friendly, direct response.
CONVERSATION_PATTERNS = [
    r"\bhi\b",
    r"\bhello\b",
    r"\bhey\b",
    r"\bgood morning\b",
    r"\bgood afternoon\b",
    r"\bgood evening\b",
    r"\bhow are you\b",
    r"\bhow are you doing\b",
    r"\bwhat's up\b",
    r"\bwhats up\b",
    r"\bthank you\b",
    r"\bthanks\b",
    r"\bbye\b",
    r"\bgoodbye\b",
    r"\bwho are you\b",
    r"\btell me a joke\b",
]

# Phrases indicating a need for current or external information.
WEB_PATTERNS = [
    r"\blatest\b",
    r"\brecent news\b",
    r"\btoday's news\b",
    r"\btoday news\b",
    r"\bright now\b",
    r"\bcurrently\b",
    r"\bcurrent price\b",
    r"\bcurrent weather\b",
    r"\bthis week\b",
    r"\bthis month\b",
    r"\b2026 updates\b",
    r"\bsearch the web\b",
    r"\bsearch online\b",
    r"\bfind online\b",
    r"\bon the internet\b",
    r"\bnews about\b",
    r"\bupdates about\b",
    r"\bwhat happened today\b",
]


def normalize_text(message):
    """Clean and normalize the user's message."""
    message = message.lower().strip()
    message = re.sub(r"\s+", " ", message)
    return message


def detect_intent(message):
    """
    Classify the message into:
    conversation, knowledge, or web_search.
    """

    text = normalize_text(message)

    # Friendly conversation detection.
    for pattern in CONVERSATION_PATTERNS:
        if re.search(pattern, text):
            return "conversation"

    # Explicit current-information/search detection.
    for pattern in WEB_PATTERNS:
        if re.search(pattern, text):
            return "web_search"

    # Use spaCy to process the sentence.
    if nlp:
        doc = nlp(text)

        lemmas = {
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct
        }

        # Current-information signals.
        current_words = {
            "latest", "recent", "today", "currently",
            "update", "news", "price", "weather"
        }

        if lemmas.intersection(current_words):
            return "web_search"

    # Default: answer using Gemini's general knowledge.
    return "knowledge"
