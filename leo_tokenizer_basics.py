
print("LEO - FIRST TOKENIZER EXPERIMENT")
print("--------------------------------")

# Tiny training corpus
sentences = [
    "leo is learning artificial intelligence",
    "leo is learning neural networks",
    "artificial intelligence is powerful",
    "neural networks learn patterns"
]

# Build vocabulary
words = []

for sentence in sentences:
    for word in sentence.split():
        if word not in words:
            words.append(word)

# Assign an ID to every word
word_to_id = {
    word: index
    for index, word in enumerate(words)
}

id_to_word = {
    index: word
    for word, index in word_to_id.items()
}

print("\nVocabulary:")
print(word_to_id)

# Convert sentence into token IDs
text = "leo is learning neural networks"

tokens = text.split()

token_ids = [
    word_to_id[word]
    for word in tokens
]

print("\nOriginal text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)

# Convert IDs back into words
decoded = [
    id_to_word[token_id]
    for token_id in token_ids
]

print("\nDecoded text:")
print(" ".join(decoded))
