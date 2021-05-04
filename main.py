import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import Constant
from TenseDataset import Word2Index, TenseDataset
from Net import TenseEncoder, TenseDecoder

if __name__ == '__main__':
    epochs = 150000
    vocab_size = 29
    hidden_size = 256
    num_layers = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_set = TenseDataset('dataset/train.txt', transform=Word2Index(15))
    train_loader = DataLoader(train_set, batch_size=8, num_workers=8, shuffle=True)

    encoder = TenseEncoder(input_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers)
    decoder = TenseDecoder(output_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers)

    encoder, decoder = encoder.to(device), decoder.to(device)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=7e-3)
    decoder_optimizer = optim.SGD(encoder.parameters(), lr=7e-3)

    criterion = nn.CrossEntropyLoss()

    kld_alpha = 0.0
    epoch_length = len(str(epochs))

    for epoch in range(epochs):

        loss = 0.0
        accuracy = 0
        kld_alpha = 0 if ((epoch % 10000) == 0) else max(kld_alpha + (1.0 / 5000), 1.0)

        for i, (word, tense) in enumerate(train_loader):
            word, tense = word.to(device), tense.to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            latent, (mu, logvar) = encoder(word, tense)
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar))

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

            ce_loss = criterion(predict.reshape(-1, predict.shape[-1]), torch.flatten(word))

            temp_loss = kld_loss * kld_alpha + ce_loss
            loss += temp_loss.item()

            y_hat = torch.argmax(predict, dim=-1).transpose(0, 1)
            accuracy += (y_hat == word).sum().item()

            temp_loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            rate = (i + 1) / len(train_loader)
            print(f'\rEpochs: {(epoch + 1):>{epoch_length}} / {epochs}, [{("=" * int(rate * 20)):<20}] {(rate * 100.0):>6.2f}%, loss: {temp_loss.item():.3f}', end='')

        loss /= len(train_loader)
        accuracy /= len(train_loader.dataset)

        print(f'\rEpochs: {(epoch + 1):>{epoch_length}} / {epochs}, [{"=" * 20}], loss: {loss:.3f}, accuracy: {accuracy:.3f}')
