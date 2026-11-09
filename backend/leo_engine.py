
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path


# ==========================================
# LEO - LANGUAGE MODEL ENGINE
# ==========================================

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

        sequence_length = inputs.shape[1]

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


# ==========================================
# LEO INFERENCE ENGINE
# ==========================================

class LeoEngine:

    def __init__(self):

        self.device = torch.device("cpu")

        project_root = Path(__file__).resolve().parent.parent

        model_path = project_root / "leo_phase3_model.pth"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}"
            )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True
        )

        self.characters = checkpoint["characters"]

        self.char_to_id = {
            character: index
            for index, character in enumerate(self.characters)
        }

        self.id_to_char = {
            index: character
            for index, character in enumerate(self.characters)
        }

        self.model = LeoLanguageModel(
            vocab_size=checkpoint["vocab_size"],
            dimension=checkpoint["dimension"],
            heads=checkpoint["heads"],
            layers=checkpoint["layers"],
            block_size=checkpoint["block_size"]
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        print("LEO model loaded successfully.")

    def generate_response(
        self,
        prompt,
        max_new_tokens=100,
        temperature=0.8
    ):

        unknown = [
            character
            for character in prompt
            if character not in self.char_to_id
        ]

        if unknown:
            raise ValueError(
                "Unsupported characters: "
                + repr("".join(sorted(set(unknown))))
            )

        if not prompt:
            raise ValueError("Message cannot be empty.")

        token_ids = [
            self.char_to_id[character]
            for character in prompt
        ]

        inputs = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device
        )

        output = self.model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        generated_ids = output[0].tolist()

        generated_text = "".join(
            self.id_to_char[index]
            for index in generated_ids
        )

        return generated_text[len(prompt):]
