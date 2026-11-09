
print("LEO - IMPROVED CHARACTER TOKENIZER")
print("-----------------------------------")

# Training text
text = """
leo is learning artificial intelligence.
leo is learning neural networks.
neural networks learn patterns.
"""

# Build character vocabulary
characters = sorted(set(text))

# Add a special unknown-character token
UNK = "<UNK>"

char_to_id = {
    character: index
    for index, character in enumerate(characters)
}

unk_id = len(char_to_id)
char_to_id[UNK] = unk_id

id_to_char = {
    index: character
    for character, index in char_to_id.items()
}

print("Vocabulary size:", len(char_to_id))

# Encode new text
new_text = "leo is learning transformers."

encoded = [
    char_to_id.get(character, unk_id)
    for character in new_text
]

print("\nOriginal text:")
print(new_text)

print("\nEncoded IDs:")
print(encoded)

# Decode
decoded = "".join(
    id_to_char[token_id]
    for token_id in encoded
)

print("\nDecoded text:")
print(decoded)

print("\nUnknown character ID:", unk_id)
print("Experiment completed!")
