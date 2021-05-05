import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

import Constant
from utils import parse
from TenseDataset import Word2Index, Index2Word, TenseTrainDataset
from Net import TenseEncoder, TenseDecoder


def belu4(predict: torch.Tensor, truth: torch.Tensor) -> float:
    y_predict = Index2Word()(predict)
    y_truth = Index2Word()(truth)

    score = 0.0
    for s1, s2 in zip(y_predict, y_truth):
        cc = SmoothingFunction()
        weights = (0.33, 0.33, 0.33) if len(truth) == 3 else (0.25, 0.25, 0.25, 0.25)

        score += sentence_bleu([s2], s1, weights=weights, smoothing_function=cc.method1)

    return (score / len(y_truth))


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    args = parse()

    max_length = args.max_length
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_set = TenseTrainDataset('dataset/train.txt', transform=Word2Index(max_length))
    train_loader = DataLoader(train_set, batch_size=32, num_workers=8, shuffle=True)

    encoder = TenseEncoder(input_size=Constant.VOCABULARY_SIZE, hidden_size=hidden_size, num_layers=num_layers)
    decoder = TenseDecoder(output_size=Constant.VOCABULARY_SIZE, hidden_size=hidden_size, num_layers=num_layers)

    encoder, decoder = encoder.to(device), decoder.to(device)

    encoder.train()
    decoder.train()

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=Constant.LR, momentum=0.9)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=Constant.LR, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    period = max(1, args.epochs // 10)
    kld_alpha = 1.0 if period == 1 else 0.0
    epoch_length = len(str(args.epochs))

    for epoch in range(args.epochs):

        kld_loss = 0.0
        ce_loss = 0.0
        belu4_score = 0.0
        accuracy = 0
        tf_rate = 0.5

        monitor = {}
        monitor_index = random.randint(0, len(train_loader) - 1)

        for i, (word, tense) in enumerate(train_loader):
            word, tense = word.to(device), tense.to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            latent, (mu, logvar) = encoder(word, tense)
            temp_kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar))
            kld_loss += temp_kld_loss.item()

            tf_rate = max(1.0 - (epoch + 1) / period, 0.5)
            enable_tf = (random.random() < tf_rate)

            input = torch.tensor([[Constant.SOS_TOKEN]]).repeat(1, word.shape[0]).to(device)
            hidden = (tense, latent)
            predict = []

            temp_ce_loss = 0.0
            for j in range(word.shape[1]):
                output, hidden = decoder(input, hidden)
                predict.append(output)

                if enable_tf:
                    input = word[:, j].reshape(1, latent.shape[1])
                else:
                    input = torch.argmax(output, dim=-1).type(torch.long)

                temp_ce_loss += criterion(output.squeeze(), word[:, j])

            ce_loss += temp_ce_loss
            predict = torch.cat(predict)

            predict = torch.argmax(predict, dim=-1).transpose(0, 1)
            accuracy += sum(np.array(Index2Word()(word)) == np.array(Index2Word()(predict)))

            (temp_kld_loss * kld_alpha + temp_ce_loss).backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            if i == monitor_index:
                monitor['word'] = word[0: 3]
                monitor['pred'] = predict[0: 3]

            temp_belu4_score = belu4(predict, word)
            belu4_score += temp_belu4_score

            rate = (i + 1) / len(train_loader)
            progress = f'Epochs: {(epoch + 1):>{epoch_length}} / {args.epochs}, [{("=" * int(rate * 20)):<20}] {(rate * 100.0):>6.2f}%'
            status = f'tf_rate: {tf_rate:.3f}, kld_alpha: {kld_alpha:.3f}'

            print(f'\r{progress}, {status}, kld_loss: {temp_kld_loss:.3f}, ce_loss: {temp_ce_loss:.3f}, belu4: {temp_belu4_score:.3f}', end='')

        kld_loss /= len(train_loader)
        ce_loss /= len(train_loader)
        belu4_score /= len(train_loader)
        accuracy /= len(train_set)

        if (epoch + 1) % 1 == 0:
            progress = f'Epochs: {(epoch + 1):>{epoch_length}} / {args.epochs}, [{("=" * 20)}]'
            status = f'tf_rate: {tf_rate:.3f}, kld_alpha: {kld_alpha:.3f}'

            print(f'\r{progress}, {status}, kld_loss: {kld_loss:.3f}, ce_loss: {ce_loss:.3f}, belu4: {belu4_score:.3f}, accuracy: {accuracy:.3f}')

        if (epoch + 1) % 10 == 0:
            monitor['word'] = Index2Word()(monitor['word'])
            monitor['pred'] = Index2Word()(monitor['pred'])

            print(f'word: {", ".join(monitor["word"])}')
            print(f'pred: {", ".join(monitor["pred"])}')

            torch.save(encoder, os.path.join(args.path, f'encoder_{epoch + 1}.weight'))
            torch.save(decoder, os.path.join(args.path, f'decoder_{epoch + 1}.weight'))

        if period == 1:
            kld_alpha = 1.0
        elif args.annealing == 'monotonic':
            kld_alpha = min(kld_alpha + (1.0 / period), 1.0)
        elif args.annealing == 'cyclical':
            kld_alpha = min(kld_alpha + (1.0 / (period / 2)), min((epoch + 1) % period, 1.0))
