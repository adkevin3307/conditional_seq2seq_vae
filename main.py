import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import Constant
from utils import parse
from TenseDataset import Word2Index, TenseTrainDataset, TenseTestDataset
from Net import TenseEncoder, TenseDecoder
import Model


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    args = parse()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_set = TenseTrainDataset('dataset/train.txt', transform=Word2Index(Constant.MAX_LENGTH))
    test_set = TenseTestDataset('dataset/test.txt', transform=Word2Index(Constant.MAX_LENGTH))

    train_loader = DataLoader(train_set, batch_size=32, num_workers=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=len(test_set), num_workers=8, shuffle=False)

    encoder = TenseEncoder(input_size=Constant.VOCABULARY_SIZE, hidden_size=args.hidden_size)
    decoder = TenseDecoder(output_size=Constant.VOCABULARY_SIZE, hidden_size=args.hidden_size)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=Constant.LR, momentum=0.9)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=Constant.LR, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    if args.trainable:
        net = {'encoder': encoder, 'decoder': decoder}
        optimizer = {'encoder_optimizer': encoder_optimizer, 'decoder_optimizer': decoder_optimizer}
        kwargs = {
            'period': args.period[0],
            'verbose_period': args.period[1],
            'save_period': args.period[2],
            'save': args.save_path,
            'annealing': args.annealing
        }

        Model.train(net, optimizer, criterion, args.epochs, train_set, train_loader, test_loader, **kwargs)

    if args.load:
        encoder = args.load[0]
        decoder = args.load[1]

    Model.evaluate_bleu4(encoder, decoder, test_loader)
    Model.evaluate_gaussian(encoder, decoder, train_set)
