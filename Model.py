import os
import random
import numpy as np
from typing import Any
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

import Constant
from TenseDataset import Index2Word


def belu4(predict: torch.Tensor, truth: torch.Tensor) -> float:
    y_predict = Index2Word()(predict)
    y_truth = Index2Word()(truth)

    score = 0.0
    for s1, s2 in zip(y_predict, y_truth):
        cc = SmoothingFunction()
        weights = (0.33, 0.33, 0.33) if len(truth) == 3 else (0.25, 0.25, 0.25, 0.25)

        score += sentence_bleu([s2], s1, weights=weights, smoothing_function=cc.method1)

    return (score / len(y_truth))


def train(net: dict, optimizer: dict, criterion: Any, epochs: int, train_loader: Any, annealing: str, path: str) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = net['encoder']
    decoder = net['decoder']
    encoder_optimizer = optimizer['encoder_optimizer']
    decoder_optimizer = optimizer['decoder_optimizer']

    encoder, decoder = encoder.to(device), decoder.to(device)

    encoder.train()
    decoder.train()

    period = max(1, epochs // 10)
    kld_alpha = 1.0 if period == 1 else 0.0
    epoch_length = len(str(epochs))

    for epoch in range(epochs):

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
            progress = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}, [{("=" * int(rate * 20)):<20}] {(rate * 100.0):>6.2f}%'
            status = f'tf_rate: {tf_rate:.3f}, kld_alpha: {kld_alpha:.3f}'

            print(f'\r{progress}, {status}, kld_loss: {temp_kld_loss:.3f}, ce_loss: {temp_ce_loss:.3f}, belu4: {temp_belu4_score:.3f}', end='')

        kld_loss /= len(train_loader)
        ce_loss /= len(train_loader)
        belu4_score /= len(train_loader)
        accuracy /= len(train_loader.dataset)

        if (epoch + 1) % 1 == 0:
            progress = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}, [{("=" * 20)}]'
            status = f'tf_rate: {tf_rate:.3f}, kld_alpha: {kld_alpha:.3f}'

            print(f'\r{progress}, {status}, kld_loss: {kld_loss:.3f}, ce_loss: {ce_loss:.3f}, belu4: {belu4_score:.3f}, accuracy: {accuracy:.3f}')

        if (epoch + 1) % 10 == 0:
            monitor['word'] = Index2Word()(monitor['word'])
            monitor['pred'] = Index2Word()(monitor['pred'])

            print(f'word: {", ".join(monitor["word"])}')
            print(f'pred: {", ".join(monitor["pred"])}')

            torch.save(encoder, os.path.join(path, f'encoder_{epoch + 1}.weight'))
            torch.save(decoder, os.path.join(path, f'decoder_{epoch + 1}.weight'))

        if period == 1:
            kld_alpha = 1.0
        elif annealing == 'monotonic':
            kld_alpha = min(kld_alpha + (1.0 / period), 1.0)
        elif annealing == 'cyclical':
            kld_alpha = min(kld_alpha + (1.0 / (period / 2)), min((epoch + 1) % period, 1.0))


def test(encoder_weight: str, decoder_weight: str, test_loader: Any) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = torch.load(encoder_weight)
    decoder = torch.load(decoder_weight)

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for x_test, y_test in test_loader:
            input_word, input_tense = x_test[0].to(device), x_test[1].to(device)
            output_word, output_tense = y_test[0].to(device), y_test[1].to(device)

            latent, _ = encoder(input_word, input_tense)

            input = torch.tensor([[1]]).repeat(1, latent.shape[1]).to(device)
            hidden = (output_tense, latent)
            predict = []

            for _ in range(latent.shape[1]):
                output, hidden = decoder(input, hidden)
                predict.append(output)

                input = torch.argmax(output, dim=-1).type(torch.long)

            predict = torch.cat(predict)
            predict = torch.argmax(predict, dim=-1).transpose(0, 1)

            input = Index2Word()(input_word)
            target = Index2Word()(output_word)
            predict = Index2Word()(predict)

            for i in range(latent.shape[1]):
                print(f'input  : {input[i]}')
                print(f'target : {target[i]}')
                print(f'predict: {predict[i]}')
