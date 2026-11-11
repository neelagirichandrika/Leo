
import re

# Load spaCy and its English model.
# If unavailable, LEO can still use rule-based detection.
try:
    import spacy

    nlp = spacy.load("en_core_web_sm")

except (ImportError, OSError):
    nlp = None
    print("spaCy unavailable. Using rule-based intent detection.")


# ==========================================
# INTENT PATTERNS
# ==========================================

CONVERSATION_PATTERNS = [
    r"\bhi\b",
    r"\bhello\b",
    r"\bhey\b",
    r"\bgood morning\b",
    r"\bgood afternoon\b",
    r"\bgood evening\b",
    r"\bgood night\b",
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

WEB_PATTERNS = [
    r"\blatest\b",
    r"\brecent news\b",
    r"\btoday's news\b",
    r"\btoday news\b",
    r"\bcurrent price\b",
    r"\bcurrent weather\b",
    r"\bthis week's news\b",
    r"\bthis month's news\b",
    r"\b2026 updates\b",
    r"\bsearch the web\b",
    r"\bsearch online\b",
    r"\bfind online\b",
    r"\bon the internet\b",
    r"\bnews about\b",
    r"\bupdates about\b",
    r"\bwhat happened today\b",
    r"\bright now\b",
    r"\bcurrently\b",
]


# ==========================================
# TEXT NORMALIZATION
# ==========================================

def normalize_text(message):

    text = message.lower().strip()

    text = re.sub(r"\s+", " ", text)

    return text


# ==========================================
# INTENT DETECTION
# ==========================================

def detect_intent(message):
    """
    Detect the user's message intent.

    Returns:
        conversation: Friendly conversation
        knowledge: General questions and explanations
        web_search: Current or external information
    """

    text = normalize_text(message)

    # 1. Check for web search intent first.
    # This prevents messages such as
    # "Hey, show me the latest news" being
    # classified as simple conversation.

    for pattern in WEB_PATTERNS:

        if re.search(pattern, text):

            return "web_search"

    # 2. Check for friendly conversation.

    for pattern in CONVERSATION_PATTERNS:

        if re.search(pattern, text):

            return "conversation"

    # 3. Process the sentence using spaCy.

    if nlp:

        doc = nlp(text)

        lemmas = {
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct
        }

        current_words = {
            "latest",
            "recent",
            "today",
            "update",
            "news",
            "price",
            "weather",
        }

        if lemmas.intersection(current_words):

            return "web_search"

    # 4. Default to general knowledge.

    return "knowledge"
