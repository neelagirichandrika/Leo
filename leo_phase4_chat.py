import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# LEO - PHASE 4
# MODEL LOADER AND CHAT PROTOTYPE
# ==========================================

device = torch.device("cpu")

MODEL_PATH = "leo_phase3_model.pth"

print("=" * 50)
print("LEO - LOADING LANGUAGE MODEL")
print("=" * 50)


# ------------------------------------------
# 1. LOAD SAVED MODEL INFORMATION
# ------------------------------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

characters = checkpoint["characters"]

vocab_size = checkpoint["vocab_size"]
block_size = checkpoint["block_size"]
dimension = checkpoint["dimension"]
heads = checkpoint["heads"]
layers = checkpoint["layers"]

char_to_id = {
    character: index
    for index, character in enumerate(characters)
}

id_to_char = {
    index: character
    for index, character in enumerate(characters)
}

print("\nModel information loaded.")
print("Vocabulary size:", vocab_size)
print("Context length:", block_size)


# ------------------------------------------
# 2. TOKENIZER
# ------------------------------------------

def encode(text):

    unknown = [
        character
        for character in text
        if character not in char_to_id
    ]

    if unknown:
        raise ValueError(
            "Unsupported characters: "
            + repr("".join(sorted(set(unknown))))
        )

    return [
        char_to_id[character]
        for character in text
    ]


def decode(token_ids):

    return "".join(
        id_to_char[index]
        for index in token_ids
    )


# ------------------------------------------
# 3. ATTENTION
# ------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, dimension, heads, dropout):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=dimension,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        sequence_length = x.shape[1]

        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        output, _ = self.attention(
            x, x, x,
            attn_mask=mask,
            need_weights=False
        )

        return self.dropout(output)


# ------------------------------------------
# 4. FEED-FORWARD NETWORK
# ------------------------------------------

class FeedForward(nn.Module):

    def __init__(self, dimension, dropout):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Linear(dimension * 4, dimension),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


# ------------------------------------------
# 5. TRANSFORMER BLOCK
# ------------------------------------------

class TransformerBlock(nn.Module):

    def __init__(self, dimension, heads, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(dimension)

        self.attention = CausalSelfAttention(
            dimension, heads, dropout
        )

        self.norm2 = nn.LayerNorm(dimension)

        self.feed_forward = FeedForward(
            dimension, dropout
        )

    def forward(self, x):

        x = x + self.attention(self.norm1(x))

        x = x + self.feed_forward(self.norm2(x))

        return x


# ------------------------------------------
# 6. RECREATE LEO'S MODEL ARCHITECTURE
# ------------------------------------------

class LeoLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        dimension,
        heads,
        layers,
        block_size,
        dropout=0.1
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size, dimension
        )

        self.position_embedding = nn.Embedding(
            block_size, dimension
        )

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    dimension, heads, dropout
                )
                for _ in range(layers)
            ]
        )

        self.final_norm = nn.LayerNorm(dimension)

        self.language_head = nn.Linear(
            dimension, vocab_size
        )

    def forward(self, inputs):

        batch, sequence_length = inputs.shape

        token_vectors = self.token_embedding(inputs)

        positions = torch.arange(
            sequence_length,
            device=inputs.device
        )

        position_vectors = self.position_embedding(
            positions
        )

        x = token_vectors + position_vectors

        x = self.transformer_blocks(x)

        x = self.final_norm(x)

        logits = self.language_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        inputs,
        max_new_tokens=150,
        temperature=0.8
    ):

        self.eval()

        for _ in range(max_new_tokens):

            context = inputs[:, -self.block_size:]

            logits = self(context)

            next_token_scores = logits[:, -1, :]

            next_token_scores = (
                next_token_scores / temperature
            )

            probabilities = F.softmax(
                next_token_scores,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            inputs = torch.cat(
                (inputs, next_token),
                dim=1
            )

        return inputs


# ------------------------------------------
# 7. LOAD TRAINED WEIGHTS
# ------------------------------------------

model = LeoLanguageModel(
    vocab_size=vocab_size,
    dimension=dimension,
    heads=heads,
    layers=layers,
    block_size=block_size
).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nTrained weights restored successfully!")


# ------------------------------------------
# 8. TEXT GENERATION FUNCTION
# ------------------------------------------

def generate_response(prompt):

    token_ids = encode(prompt)

    if not token_ids:
        return ""

    inputs = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device
    )

    output = model.generate(
        inputs,
        max_new_tokens=150,
        temperature=0.8
    )

    generated_ids = output[0].tolist()

    return decode(generated_ids)


# ------------------------------------------
# 9. LEO TERMINAL CHAT
# ------------------------------------------

print("\n" + "=" * 50)
print("LEO AI ASSISTANT")
print("=" * 50)

print("Type a message to generate text.")
print("Type exit to close LEO.")

while True:

    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("\nLEO: Goodbye!")
        break

    try:

        generated_text = generate_response(
            user_input
        )

        response = generated_text[len(user_input):]

        print("\nLEO:", response)

    except ValueError as error:

        print("\nLEO:", error)

    except Exception as error:

        print("\nUnexpected error:", error)