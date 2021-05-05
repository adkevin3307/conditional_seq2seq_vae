import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

import Constant
from TenseDataset import Word2Index, Index2Word, TenseDataset
from Net import TenseEncoder, TenseDecoder


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--max_length', type=int, default=15)
    parser.add_argument('-s', '--size', type=int, dest='hidden_size', default=256)
    parser.add_argument('-n', '--num_layers', type=int, default=2)
    parser.add_argument('-a', '--annealing', type=str, default='cyclical', choices=['monotonic', 'cyclical'])

    args = parser.parse_args()

    print('=' * 50)
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('=' * 50)

    return args


def belu4(predict: torch.Tensor, truth: torch.Tensor) -> float:
    y_predict = Index2Word()(predict)
    y_truth = Index2Word()(truth)

    score = 0.0
    for y_hat, y in zip(y_predict, y_truth):
        cc = SmoothingFunction()
        weights = (0.33, 0.33, 0.33) if len(truth) == 3 else (0.25, 0.25, 0.25, 0.25)

        score += sentence_bleu([y], y_hat, weights=weights, smoothing_function=cc.method1)

    return score


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

    train_set = TenseDataset('dataset/train.txt', transform=Word2Index(max_length))
    train_loader = DataLoader(train_set, batch_size=32, num_workers=8, shuffle=True)

    encoder = TenseEncoder(input_size=Constant.VOCABULARY_SIZE, hidden_size=hidden_size, num_layers=num_layers)
    decoder = TenseDecoder(output_size=Constant.VOCABULARY_SIZE, hidden_size=hidden_size, num_layers=num_layers, max_length=max_length)

    encoder, decoder = encoder.to(device), decoder.to(device)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=Constant.LR)
    decoder_optimizer = optim.SGD(encoder.parameters(), lr=Constant.LR)

    criterion = nn.CrossEntropyLoss()

    kld_alpha = 0.0
    epoch_length = len(str(Constant.EPOCHS))

    for epoch in range(Constant.EPOCHS):

        kld_loss = 0.0
        ce_loss = 0.0
        belu4_score = 0.0
        accuracy = 0

        monitor = {}
        monitor_index = random.randint(0, len(train_loader))

        for i, (word, tense) in enumerate(train_loader):
            word, tense = word.to(device), tense.to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            latent, (mu, logvar) = encoder(word, tense)
            temp_kld_loss = kld_alpha * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar)))

            enable_teacher_forcing = (random.random() < 0.5)

            input = torch.tensor([[Constant.SOS_TOKEN]]).repeat(1, latent.shape[1]).to(device)
            hidden = (tense, latent)
            predict = []

            for j in range(word.shape[1]):
                output, hidden = decoder(input, hidden)
                predict.append(output)

                if enable_teacher_forcing:
                    input = word[:, j].reshape(1, latent.shape[1])
                else:
                    input = torch.argmax(output, dim=-1).type(torch.long)

            predict = torch.cat(predict)

            temp_ce_loss = criterion(predict.reshape(-1, predict.shape[-1]), torch.flatten(word))

            temp_loss = temp_kld_loss + temp_ce_loss
            kld_loss += temp_kld_loss.item()
            ce_loss += temp_ce_loss.item()

            predict = torch.argmax(predict, dim=-1).transpose(0, 1)
            accuracy += (predict == word).sum().item()

            if i == monitor_index:
                monitor['word'] = np.squeeze(word.cpu().numpy())[0]
                monitor['predict'] = np.squeeze(predict.cpu().numpy())[0]

            temp_loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            temp_belu4_score = belu4(predict, word)
            belu4_score += temp_belu4_score

            rate = (i + 1) / len(train_loader)
            progress = f'Epochs: {(epoch + 1):>{epoch_length}} / {Constant.EPOCHS}, [{("=" * int(rate * 20)):<20}] {(rate * 100.0):>6.2f}%'
            print(f'\r{progress}, kld_loss: {temp_kld_loss.item():.3f}, ce_loss: {temp_ce_loss.item():.3f}, belu4: {temp_belu4_score:.3f}', end='')

        kld_loss /= len(train_loader)
        ce_loss /= len(train_loader)
        belu4_score /= len(train_loader)
        accuracy /= len(train_set)

        if args.annealing == 'monotonic':
            kld_alpha = min(kld_alpha + (1.0 / 10000), 1.0)
        elif args.annealing == 'cyclical':
            kld_alpha = 0 if (((epoch + 1) % 10000) == 0) else min(kld_alpha + (1.0 / 5000), 1.0)

        if (epoch + 1) % 100 == 0:
            progress = f'Epochs: {(epoch + 1):>{epoch_length}} / {Constant.EPOCHS}, [{("=" * 20)}]'
            print(f'\r{progress}, kld_loss: {kld_loss:.3f}, ce_loss: {ce_loss:.3f}, belu4: {belu4_score:.3f}, accuracy: {accuracy:.3f}')

        if (epoch + 1) % 1000 == 0:
            print(f'   word: {monitor["word"]}')
            print(f'predict: {monitor["predict"]}')

            torch.save(encoder, f'weights/encoder_{epoch + 1}.weight')
            torch.save(decoder, f'weights/decoder_{epoch + 1}.weight')
