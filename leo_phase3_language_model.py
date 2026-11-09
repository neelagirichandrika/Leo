import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# LEO - PHASE 3
# FIRST TRAINABLE LANGUAGE MODEL
# ==========================================

torch.manual_seed(42)

device = torch.device("cpu")

print("=" * 55)
print("LEO - FIRST LANGUAGE MODEL")
print("=" * 55)

# ------------------------------------------
# 1. TRAINING DATA
# ------------------------------------------

text = """
LEO is an artificial intelligence project.
LEO learns patterns from text.
A language model learns to predict the next token.
A transformer uses attention to understand context.
Machine learning improves a model through training.
LEO is learning how language works.
Artificial intelligence can process and generate text.
A neural network learns by adjusting its parameters.
LEO will become a helpful AI assistant.
""" * 40

# ------------------------------------------
# 2. CHARACTER TOKENIZER
# ------------------------------------------

characters = sorted(set(text))

vocab_size = len(characters)

char_to_id = {
    character: index
    for index, character in enumerate(characters)
}

id_to_char = {
    index: character
    for index, character in enumerate(characters)
}

def encode(sentence):
    return [char_to_id[character] for character in sentence]

def decode(token_ids):
    return "".join(id_to_char[index] for index in token_ids)

data = torch.tensor(encode(text), dtype=torch.long)

print("\n1. TOKENIZER")
print("Vocabulary size:", vocab_size)
print("Total characters:", len(data))

# ------------------------------------------
# 3. TRAINING AND VALIDATION DATA
# ------------------------------------------

split_index = int(0.9 * len(data))

train_data = data[:split_index]
validation_data = data[split_index:]

block_size = 64
batch_size = 32

def get_batch(source):
    positions = torch.randint(
        len(source) - block_size - 1,
        (batch_size,)
    )

    inputs = torch.stack([
        source[position:position + block_size]
        for position in positions
    ])

    targets = torch.stack([
        source[position + 1:position + block_size + 1]
        for position in positions
    ])

    return inputs.to(device), targets.to(device)

print("\n2. DATASET")
print("Training characters:", len(train_data))
print("Validation characters:", len(validation_data))
print("Context length:", block_size)

# ------------------------------------------
# 4. CAUSAL SELF-ATTENTION
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
# 5. FEED-FORWARD NETWORK
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
# 6. TRANSFORMER BLOCK
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
# 7. COMPLETE LEO LANGUAGE MODEL
# ------------------------------------------

class LeoLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        dimension=64,
        heads=4,
        layers=2,
        block_size=64,
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
                TransformerBlock(dimension, heads, dropout)
                for _ in range(layers)
            ]
        )

        self.final_norm = nn.LayerNorm(dimension)

        self.language_head = nn.Linear(
            dimension, vocab_size
        )

    def forward(self, inputs, targets=None):

        batch, sequence_length = inputs.shape

        token_vectors = self.token_embedding(inputs)

        positions = torch.arange(
            sequence_length,
            device=inputs.device
        )

        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors

        x = self.transformer_blocks(x)

        x = self.final_norm(x)

        logits = self.language_head(x)

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, inputs, max_new_tokens=150):

        self.eval()

        for _ in range(max_new_tokens):

            context = inputs[:, -self.block_size:]

            logits, _ = self(context)

            next_token_scores = logits[:, -1, :]

            probabilities = F.softmax(
                next_token_scores, dim=-1
            )

            next_token = torch.multinomial(
                probabilities, num_samples=1
            )

            inputs = torch.cat(
                (inputs, next_token), dim=1
            )

        return inputs

# ------------------------------------------
# 8. CREATE LEO
# ------------------------------------------

model = LeoLanguageModel(
    vocab_size=vocab_size,
    dimension=64,
    heads=4,
    layers=2,
    block_size=block_size,
    dropout=0.1
).to(device)

print("\n3. MODEL")
print("Parameters:", sum(
    parameter.numel()
    for parameter in model.parameters()
))

# ------------------------------------------
# 9. OPTIMIZER
# ------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

# ------------------------------------------
# 10. LOSS EVALUATION
# ------------------------------------------

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for name, source in [
        ("train", train_data),
        ("validation", validation_data)
    ]:

        losses = []

        for _ in range(10):

            inputs, targets = get_batch(source)

            _, loss = model(inputs, targets)

            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()

    return results

# ------------------------------------------
# 11. TRAIN LEO
# ------------------------------------------

print("\n4. TRAINING STARTED")

iterations = 500

for step in range(iterations):

    inputs, targets = get_batch(train_data)

    logits, loss = model(inputs, targets)

    optimizer.zero_grad(set_to_none=True)

    loss.backward()

    optimizer.step()

    if step % 100 == 0:

        losses = estimate_loss()

        print(
            f"Step {step:4d} | "
            f"Training Loss: {losses['train']:.4f} | "
            f"Validation Loss: {losses['validation']:.4f}"
        )

print("\nTRAINING COMPLETED!")

# ------------------------------------------
# 12. GENERATE TEXT
# ------------------------------------------

print("\n5. LEO TEXT GENERATION")

starting_text = "LEO"

starting_ids = torch.tensor(
    [encode(starting_text)],
    dtype=torch.long,
    device=device
)

generated_ids = model.generate(
    starting_ids,
    max_new_tokens=150
)

generated_text = decode(
    generated_ids[0].tolist()
)

print("\nGenerated by LEO:")
print(generated_text)

# ------------------------------------------
# 13. SAVE TRAINED MODEL
# ------------------------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "characters": characters,
        "block_size": block_size,
        "dimension": 64,
        "heads": 4,
        "layers": 2
    },
    "leo_phase3_model.pth"
)

print("\nModel saved as leo_phase3_model.pth")

print("\n" + "=" * 55)
print("LEO PHASE 3 COMPLETED")
print("=" * 55)