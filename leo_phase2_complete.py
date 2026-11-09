
import torch
import torch.nn as nn
import math

torch.manual_seed(42)

print("=" * 50)
print("LEO - COMPLETE PHASE 2")
print("LANGUAGE MODEL FOUNDATIONS")
print("=" * 50)

# ------------------------------------------
# 1. TOKEN IDS
# ------------------------------------------

# These IDs come from our small BPE experiment.
token_ids = torch.tensor([[6, 8, 11, 2, 17]])

vocab_size = 20
embedding_dim = 16
sequence_length = token_ids.shape[1]

print("\n1. TOKEN IDS")
print("Token IDs:", token_ids)
print("Sequence length:", sequence_length)


# ------------------------------------------
# 2. TOKEN EMBEDDINGS
# ------------------------------------------

token_embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)

embedded_tokens = token_embedding(token_ids)

print("\n2. TOKEN EMBEDDINGS")
print("Embedding shape:", embedded_tokens.shape)


# ------------------------------------------
# 3. POSITIONAL EMBEDDINGS
# ------------------------------------------

position_embedding = nn.Embedding(
    num_embeddings=32,
    embedding_dim=embedding_dim
)

positions = torch.arange(sequence_length)

position_vectors = position_embedding(positions)

position_vectors = position_vectors.unsqueeze(0)

# Combine token and position information
x = embedded_tokens + position_vectors

print("\n3. POSITIONAL EMBEDDINGS")
print("Position shape:", position_vectors.shape)
print("Combined shape:", x.shape)


# ------------------------------------------
# 4. QUERY, KEY AND VALUE
# ------------------------------------------

query_layer = nn.Linear(embedding_dim, embedding_dim)
key_layer = nn.Linear(embedding_dim, embedding_dim)
value_layer = nn.Linear(embedding_dim, embedding_dim)

Q = query_layer(x)
K = key_layer(x)
V = value_layer(x)

print("\n4. QUERY, KEY AND VALUE")
print("Query shape:", Q.shape)
print("Key shape:", K.shape)
print("Value shape:", V.shape)


# ------------------------------------------
# 5. SCALED DOT-PRODUCT ATTENTION
# ------------------------------------------

scores = Q @ K.transpose(-2, -1)

scores = scores / math.sqrt(embedding_dim)

# Causal mask: prevent access to future tokens
causal_mask = torch.triu(
    torch.ones(
        sequence_length,
        sequence_length,
        dtype=torch.bool
    ),
    diagonal=1
)

scores = scores.masked_fill(
    causal_mask,
    float("-inf")
)

attention_weights = torch.softmax(scores, dim=-1)

attention_output = attention_weights @ V

print("\n5. CAUSAL SELF-ATTENTION")
print("Attention weights:", attention_weights.shape)
print("Attention output:", attention_output.shape)


# ------------------------------------------
# 6. MULTI-HEAD ATTENTION
# ------------------------------------------

multi_head_attention = nn.MultiheadAttention(
    embed_dim=embedding_dim,
    num_heads=4,
    batch_first=True
)

multi_head_output, head_weights = multi_head_attention(
    x,
    x,
    x,
    attn_mask=causal_mask,
    need_weights=True,
    average_attn_weights=False
)

print("\n6. MULTI-HEAD ATTENTION")
print("Output shape:", multi_head_output.shape)
print("Head weights shape:", head_weights.shape)


# ------------------------------------------
# 7. FEED-FORWARD NETWORK
# ------------------------------------------

feed_forward = nn.Sequential(
    nn.Linear(embedding_dim, 64),
    nn.GELU(),
    nn.Linear(64, embedding_dim)
)

feed_forward_output = feed_forward(multi_head_output)

print("\n7. FEED-FORWARD NETWORK")
print("Output shape:", feed_forward_output.shape)


# ------------------------------------------
# 8. RESIDUAL CONNECTIONS
# ------------------------------------------

residual_output = x + multi_head_output

print("\n8. RESIDUAL CONNECTION")
print("Output shape:", residual_output.shape)


# ------------------------------------------
# 9. LAYER NORMALIZATION
# ------------------------------------------

layer_norm = nn.LayerNorm(embedding_dim)

normalized_output = layer_norm(residual_output)

print("\n9. LAYER NORMALIZATION")
print("Output shape:", normalized_output.shape)


# ------------------------------------------
# 10. COMPLETE TRANSFORMER BLOCK
# ------------------------------------------

class LeoTransformerBlock(nn.Module):

    def __init__(self, dimension, heads):

        super().__init__()

        self.attention = nn.MultiheadAttention(
            dimension,
            heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dimension)

        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.GELU(),
            nn.Linear(dimension * 4, dimension)
        )

        self.norm2 = nn.LayerNorm(dimension)

    def forward(self, x):

        length = x.shape[1]

        mask = torch.triu(
            torch.ones(
                length,
                length,
                dtype=torch.bool,
                device=x.device
            ),
            diagonal=1
        )

        attention_output, _ = self.attention(
            x, x, x,
            attn_mask=mask,
            need_weights=False
        )

        x = self.norm1(x + attention_output)

        feed_output = self.feed_forward(x)

        x = self.norm2(x + feed_output)

        return x


transformer = LeoTransformerBlock(
    dimension=embedding_dim,
    heads=4
)

transformer_output = transformer(x)

print("\n10. COMPLETE TRANSFORMER BLOCK")
print("Transformer output:", transformer_output.shape)


# ------------------------------------------
# 11. NEXT-TOKEN PREDICTION SCORES
# ------------------------------------------

language_head = nn.Linear(
    embedding_dim,
    vocab_size
)

logits = language_head(transformer_output)

print("\n11. LANGUAGE MODEL OUTPUT")
print("Logits shape:", logits.shape)

print("\nLEO PHASE 2 DEMONSTRATION COMPLETED!")
print("=" * 50)
