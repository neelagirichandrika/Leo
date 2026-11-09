
print("LEO - SUBWORD TOKENIZER")
print("------------------------")

# Reusable word pieces
vocabulary = [
    "learn",
    "ing",
    "ed",
    "er",
    "s",
    "art",
    "ificial",
    "intelligence",
    "neural",
    "network"
]

# Longest pieces first
vocabulary = sorted(vocabulary, key=len, reverse=True)


def tokenize_word(word):
    tokens = []
    position = 0

    while position < len(word):
        matched = False

        for piece in vocabulary:
            if word.startswith(piece, position):
                tokens.append(piece)
                position += len(piece)
                matched = True
                break

        if not matched:
            tokens.append("<UNK>")
            position += 1

    return tokens


words = [
    "learning",
    "learned",
    "learner",
    "learns",
    "artificial",
    "neural"
]

for word in words:
    tokens = tokenize_word(word)

    print("\nWord:", word)
    print("Subwords:", tokens)
