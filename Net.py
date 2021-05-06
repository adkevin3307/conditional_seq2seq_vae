import torch
import torch.nn as nn

import Constant


class TenseEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super(TenseEncoder, self).__init__()

        self.word_embedding = nn.Embedding(input_size, hidden_size)
        self.condition_embedding = nn.Embedding(Constant.CONDITION_CATEGORY, Constant.CONDITION_EMBEDDING_SIZE)

        self.lstm = nn.LSTM(hidden_size, hidden_size, Constant.NUM_LAYERS, bidirectional=Constant.BIDIRECTIONAL)

        self.linear_1 = nn.Linear(hidden_size, Constant.LATENT_SIZE)
        self.linear_2 = nn.Linear(hidden_size, Constant.LATENT_SIZE)

    def forward(self, input: torch.Tensor, condition: torch.Tensor) -> tuple:
        output = self.word_embedding(input).transpose(0, 1)
        hidden = self._hidden(condition)

        output, hidden = self.lstm(output, (hidden, hidden))

        mu = self.linear_1(hidden[0])
        logvar = self.linear_2(hidden[0])
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        latent = mu + eps * std

        return (latent, (mu, logvar))

    def _hidden(self, condition: torch.Tensor) -> torch.Tensor:
        repeat = Constant.NUM_LAYERS * (2 if Constant.BIDIRECTIONAL else 1)

        hidden = torch.zeros(repeat, condition.shape[0], self.lstm.hidden_size - Constant.CONDITION_EMBEDDING_SIZE).to(condition.device)
        condition = self.condition_embedding(condition).repeat(repeat, 1, 1)

        hidden = torch.cat([hidden, condition], dim=-1)

        return hidden


class TenseDecoder(nn.Module):
    def __init__(self, output_size: int, hidden_size: int) -> None:
        super(TenseDecoder, self).__init__()

        self.word_embedding = nn.Embedding(output_size, hidden_size)
        self.condition_embedding = nn.Embedding(Constant.CONDITION_CATEGORY, Constant.CONDITION_EMBEDDING_SIZE)

        self.lstm = nn.LSTM(hidden_size, hidden_size, Constant.NUM_LAYERS, bidirectional=Constant.BIDIRECTIONAL)

        self.linear_1 = nn.Linear(Constant.LATENT_SIZE + Constant.CONDITION_EMBEDDING_SIZE, hidden_size)
        self.linear_2 = nn.Linear(hidden_size * (2 if Constant.BIDIRECTIONAL else 1), output_size)

    def forward(self, input: torch.Tensor, hidden: tuple) -> tuple:
        output = self.word_embedding(input)

        if hidden[0].shape != hidden[1].shape:
            condition, latent = hidden[0], hidden[1]
            hidden = self.linear_1(self._hidden(condition, latent))
            hidden = (hidden, hidden)

        output, hidden = self.lstm(output, hidden)
        output = self.linear_2(output)

        return (output, hidden)

    def _hidden(self, condition: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        repeat = Constant.NUM_LAYERS * (2 if Constant.BIDIRECTIONAL else 1)

        condition = self.condition_embedding(condition).repeat(repeat, 1, 1)

        hidden = torch.cat([latent, condition], dim=-1)

        return hidden
