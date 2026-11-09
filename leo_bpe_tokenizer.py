
from collections import Counter

print("LEO - BYTE PAIR ENCODING")
print("------------------------")

# Small training corpus
corpus = [
    "low",
    "lower",
    "lowest",
    "low",
    "lower",
    "new",
    "newer",
    "newest"
]

# Initially, represent every word as characters
words = [
    list(word) + ["</w>"]
    for word in corpus
]


def count_pairs(words):
    pairs = Counter()

    for word in words:
        for i in range(len(word) - 1):
            pairs[(word[i], word[i + 1])] += 1

    return pairs


def merge_pair(words, pair):
    merged_words = []

    for word in words:
        new_word = []
        i = 0

        while i < len(word):

            if (
                i < len(word) - 1
                and (word[i], word[i + 1]) == pair
            ):
                new_word.append(word[i] + word[i + 1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        merged_words.append(new_word)

    return merged_words


# Learn merges
num_merges = 10
learned_merges = []

for step in range(num_merges):

    pairs = count_pairs(words)

    if not pairs:
        break

    best_pair, frequency = pairs.most_common(1)[0]

    words = merge_pair(words, best_pair)

    learned_merges.append(best_pair)

    print(
        f"Merge {step + 1}: "
        f"{best_pair} | Frequency: {frequency}"
    )


print("\nLEARNED MERGES")
print("------------------------")

for merge in learned_merges:
    print(merge)

print("\nFINAL TOKENIZED CORPUS")

for word in words:
    print(word)
