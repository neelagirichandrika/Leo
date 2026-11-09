
from collections import Counter

print("LEO - BPE TOKEN IDs")
print("-------------------")

# Training corpus
corpus = [
    "low", "lower", "lowest", "low",
    "lower", "new", "newer", "newest"
]

# Convert words into characters
words = [list(word) + ["</w>"] for word in corpus]

# Store initial symbols
symbols = set()

for word in words:
    symbols.update(word)

# Learn BPE merges
learned_merges = []

for step in range(10):

    pairs = Counter()

    for word in words:
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1

    if not pairs:
        break

    best_pair = pairs.most_common(1)[0][0]

    # Merge the selected pair
    merged_words = []

    for word in words:
        new_word = []
        i = 0

        while i < len(word):
            if (
                i < len(word) - 1
                and (word[i], word[i + 1]) == best_pair
            ):
                merged = word[i] + word[i + 1]
                new_word.append(merged)
                symbols.add(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        merged_words.append(new_word)

    words = merged_words
    learned_merges.append(best_pair)

# Create token vocabulary
symbols.add("<UNK>")

token_to_id = {
    token: index
    for index, token in enumerate(sorted(symbols))
}

id_to_token = {
    index: token
    for token, index in token_to_id.items()
}

print("Vocabulary size:", len(token_to_id))


# Apply learned merges to a new word
def encode_word(word):

    tokens = list(word) + ["</w>"]

    for pair in learned_merges:

        merged_tokens = []
        i = 0

        while i < len(tokens):

            if (
                i < len(tokens) - 1
                and (tokens[i], tokens[i + 1]) == pair
            ):
                merged_tokens.append(tokens[i] + tokens[i + 1])
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1

        tokens = merged_tokens

    return [
        token if token in token_to_id else "<UNK>"
        for token in tokens
    ]


# Encode a sentence
sentence = "low lower newest"

encoded_tokens = []

for word in sentence.split():
    encoded_tokens.extend(encode_word(word))

token_ids = [
    token_to_id[token]
    for token in encoded_tokens
]

print("\nOriginal sentence:")
print(sentence)

print("\nBPE tokens:")
print(encoded_tokens)

print("\nToken IDs:")
print(token_ids)

# Decode
decoded_tokens = [
    id_to_token[token_id]
    for token_id in token_ids
]

decoded_text = "".join(decoded_tokens)
decoded_text = decoded_text.replace("</w>", " ")

print("\nDecoded sentence:")
print(decoded_text.strip())

print("\nExperiment completed!")
