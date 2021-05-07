import os
import math
import random
import logging
import numpy as np
from typing import Any
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

import Constant
from TenseDataset import Index2Word, TenseTrainDataset

logger = logging.getLogger('__main__.Model')


def bleu4(predict: torch.Tensor, truth: torch.Tensor) -> float:
    y_predict = Index2Word()(predict)
    y_truth = Index2Word()(truth)

    score = 0.0
    for s1, s2 in zip(y_predict, y_truth):
        cc = SmoothingFunction()
        weights = (0.33, 0.33, 0.33) if len(truth) == 3 else (0.25, 0.25, 0.25, 0.25)

        score += sentence_bleu([s2], s1, weights=weights, smoothing_function=cc.method1)

    return (score / len(y_truth))


def evaluate_bleu4(encoder: Any, decoder: Any, test_loader: Any, verbose: bool = True) -> float:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if isinstance(encoder, str):
        encoder = torch.load(encoder)

    if isinstance(decoder, str):
        decoder = torch.load(decoder)

    encoder.eval()
    decoder.eval()

    inputs = []
    targets = []
    predicts = []
    accuracy = 0
    bleu4_score = 0.0

    with torch.no_grad():
        for x_test, y_test in test_loader:
            input_word, input_tense = x_test[0].to(device), x_test[1].to(device)
            output_word, output_tense = y_test[0].to(device), y_test[1].to(device)

            latent, _ = encoder(input_word, input_tense)

            input = torch.tensor([[Constant.SOS_TOKEN]]).repeat(1, test_loader.batch_size).to(device)
            hidden = (output_tense, latent)
            predict = []

            for _ in range(test_loader.batch_size):
                output, hidden = decoder(input, hidden)
                predict.append(output)

                input = torch.argmax(output, dim=-1).type(torch.long)

            predict = torch.cat(predict)
            predict = torch.argmax(predict, dim=-1).transpose(0, 1)

            accuracy += sum(np.array(Index2Word()(output_word)) == np.array(Index2Word()(predict)))

            temp_bleu4_score = bleu4(predict, output_word)
            bleu4_score += temp_bleu4_score

            inputs.extend(Index2Word()(input_word))
            targets.extend(Index2Word()(output_word))
            predicts.extend(Index2Word()(predict))

    accuracy /= len(test_loader.dataset)
    bleu4_score /= len(test_loader)

    if verbose:
        print('=' * 50)

        for i in range(test_loader.batch_size):
            logger.info(f'input  : {inputs[i]}')
            logger.info(f'target : {targets[i]}')
            logger.info(f'predict: {predicts[i]}')

            if i < (test_loader.batch_size - 1):
                print()

        print('-' * 50)

        logger.info(f'accuracy: {accuracy:.3f}, bleu4: {bleu4_score:.3f}')

        print('=' * 50)

    return bleu4_score


def generate_gaussian_data(n: int) -> tuple:
    latents = torch.empty(Constant.NUM_LAYERS * (2 if Constant.BIDIRECTIONAL else 1), n, Constant.LATENT_SIZE)
    latents = latents.normal_(mean=0, std=1)
    latents = torch.repeat_interleave(latents, Constant.CONDITION_CATEGORY, dim=1)

    tenses = torch.tensor(list(range(4))).repeat(n)

    return (latents, tenses)


def evaluate_gaussian(encoder: Any, decoder: Any, train_set: TenseTrainDataset, verbose: bool = True) -> float:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if isinstance(encoder, str):
        encoder = torch.load(encoder)

    if isinstance(decoder, str):
        decoder = torch.load(decoder)

    encoder.eval()
    decoder.eval()

    latents, tenses = generate_gaussian_data(100)
    latents, tenses = latents.to(device), tenses.to(device)

    input = torch.tensor([[Constant.SOS_TOKEN]]).repeat(1, latents.shape[1]).to(device)
    hidden = (tenses, latents)

    predicts = []

    for _ in range(Constant.MAX_LENGTH + 2):
        output, hidden = decoder(input, hidden)
        predicts.append(output)

        input = torch.argmax(output, dim=-1).type(torch.long)

    predicts = torch.cat(predicts)
    predicts = torch.argmax(predicts, dim=-1).transpose(0, 1)
    predicts = Index2Word()(predicts)

    score = 0
    words = []
    corrects = []

    for i in range(len(train_set)):
        word = train_set[i][0].reshape(1, -1)
        word = Index2Word()(word)

        words.extend(word)

    predicts = [predicts[i: (i + 4)] for i in range(0, len(predicts), 4)]
    words = [words[i: (i + 4)] for i in range(0, len(words), 4)]

    for word in words:
        for predict in predicts:
            if word == predict:
                corrects.append(word)
                score += 1

    score /= len(predicts)

    if verbose:
        print('=' * 50)
        logger.info(f'predict words: {len(predicts)}')
        print('-' * 50)

        for correct in corrects:
            logger.info(correct)

        print('-' * 50)
        logger.info(f'gaussian score: {score:.3f}')
        print('=' * 50)

    return score


def train(net: dict, optimizer: dict, criterion: Any, epochs: int, train_set: Any, train_loader: Any, test_loader: Any, **kwargs) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = net['encoder']
    decoder = net['decoder']
    encoder_optimizer = optimizer['encoder_optimizer']
    decoder_optimizer = optimizer['decoder_optimizer']

    encoder, decoder = encoder.to(device), decoder.to(device)

    period = kwargs['period']
    verbose_period = kwargs['verbose_period']
    save_period = kwargs['save_period']

    path = kwargs['save']
    annealing = kwargs['annealing']
    kld_alpha = 1.0 if period == 1 else 0.0

    epoch_length = len(str(epochs))

    for epoch in range(epochs):
        encoder.train()
        decoder.train()

        kld_loss = 0.0
        ce_loss = 0.0
        bleu4_score = 0.0
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

            tf_rate = (math.sin(epoch) / epoch + 0.5) if epoch else 1.0
            tf_rate = max(min(tf_rate, 1.0), 0.0)
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

            (temp_kld_loss * kld_alpha + temp_ce_loss).backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            if i == monitor_index:
                monitor['word'] = word[0: 5]
                monitor['pred'] = predict[0: 5]

            temp_bleu4_score = bleu4(predict, word)
            bleu4_score += temp_bleu4_score

            rate = (i + 1) / len(train_loader)
            progress = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}, [{("=" * int(rate * 20)):<20}] {(rate * 100.0):>6.2f}%'
            status = f'tf_rate: {tf_rate:.3f}, kld_alpha: {kld_alpha:.3f}'

            print(f'\r{progress}, {status}, kld_loss: {temp_kld_loss:.3f}, ce_loss: {temp_ce_loss:.3f}, bleu4: {temp_bleu4_score:.3f}', end='')

        kld_loss /= len(train_loader)
        ce_loss /= len(train_loader)
        bleu4_score /= len(train_loader)
        test_bleu4_score = evaluate_bleu4(encoder, decoder, test_loader, verbose=False)
        gaussian_score = evaluate_gaussian(encoder, decoder, train_set, verbose=False)

        logger.debug(f'Epoch: {epoch + 1}')
        logger.debug(f'tf_rate: {tf_rate}, kld_alpha: {kld_alpha}')
        logger.debug(f'kld_loss: {kld_loss}, ce_loss: {ce_loss}, bleu4: {bleu4_score}')
        logger.debug(f'test_bleu4: {test_bleu4_score}, gaussian: {gaussian_score}')

        if (epoch + 1) % verbose_period == 0:
            progress = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}, [{("=" * 20)}]'
            status = f'tf_rate: {tf_rate:.3f}, kld_alpha: {kld_alpha:.3f}'
            additional = f'test_bleu4: {test_bleu4_score:.3f}, gaussian: {gaussian_score:.3f}'

            print(f'\r{progress}, {status}, kld_loss: {kld_loss:.3f}, ce_loss: {ce_loss:.3f}, bleu4: {bleu4_score:.3f}, {additional}')

        if (epoch + 1) % save_period == 0:
            monitor['word'] = Index2Word()(monitor['word'])
            monitor['pred'] = Index2Word()(monitor['pred'])

            print(f'word: {", ".join(monitor["word"])}')
            print(f'pred: {", ".join(monitor["pred"])}')

            torch.save(encoder, os.path.join(path, f'encoder_{epoch + 1}.weight'))
            torch.save(decoder, os.path.join(path, f'decoder_{epoch + 1}.weight'))

        if period == 1:
            kld_alpha = 1.0
        elif annealing == 'monotonic':
            kld_alpha = min((epoch + 1) / period, 1.0)
        elif annealing == 'cyclical':
            kld_alpha = min(((epoch + 1) % period) / (period / 2), 1.0)
