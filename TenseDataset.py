from typing import Callable
import torch
from torch.utils.data import Dataset


class Word2Index(object):
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length

    def __call__(self, sample: str) -> torch.Tensor:
        result = []

        for c in sample:
            result.append(ord(c.lower()) - ord('a'))

        while len(result) < self.max_length:
            result.append(0)

        return torch.tensor(result)


class TenseDataset(Dataset):
    def __init__(self, path: str, transform: Callable = None) -> None:
        self.path = path
        self.transform = transform

        self.data = []
        with open(self.path, 'r') as txt_file:
            for line in txt_file:
                self.data.append(line.split())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[str, int]:
        row = index % len(self.data)
        col = index // len(self.data)

        word = self.data[row][col]

        if self.transform:
            word = self.transform(word)

        return (word, col)
