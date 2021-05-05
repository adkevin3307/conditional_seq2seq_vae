from typing import Callable
import torch
from torch.utils.data import Dataset

import Constant


class Word2Index(object):
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length

    def __call__(self, sample: str) -> torch.Tensor:
        result = []

        result += [Constant.SOS_TOKEN]
        result += [ord(c) - ord('a') + Constant.ALP_TOKEN for c in sample.lower()]
        result += [Constant.EOS_TOKEN]
        result += ([Constant.PAD_TOKEN] * (self.max_length - len(sample)))

        return torch.tensor(result)


class Index2Word(object):
    def __call__(self, sample: torch.Tensor) -> list:
        result = []

        sample = sample.cpu().numpy()

        for word_list in sample:
            word = ''

            for c in word_list[1: -1]:
                if c == 2:
                    break

                word += chr(c - Constant.ALP_TOKEN + ord('a'))

            result.append(word)

        return result


class TenseDataset(Dataset):
    def __init__(self, path: str, transform: Callable = None) -> None:
        self.path = path
        self.transform = transform

        self.data = []
        with open(self.path, 'r') as txt_file:
            for line in txt_file:
                self.data.extend(line.split())

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        word = self.data[index]
        tense = index % 4

        if self.transform:
            word = self.transform(word)

        return (word, tense)
