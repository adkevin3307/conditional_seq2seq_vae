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


class TenseTrainDataset(Dataset):
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


class TenseTestDataset(Dataset):
    def __init__(self, path: str, transform: Callable = None) -> None:
        self.path = path
        self.transform = transform

        self.data = []
        self.target = []
        with open(self.path, 'r') as txt_file:
            for line in txt_file:
                token = line.split()

                self.data.append([token[0], token[2]])
                self.target.append([token[1], token[3]])

    def __len__(self) -> int:
        assert len(self.data) == len(self.target)

        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        input_word, input_tense = self.data[index]
        output_word, output_tense = self.target[index]

        input_tense = int(input_tense)
        output_tense = int(output_tense)

        if self.transform:
            input_word = self.transform(input_word)
            output_word = self.transform(output_word)

        return ((input_word, input_tense), (output_word, output_tense))
