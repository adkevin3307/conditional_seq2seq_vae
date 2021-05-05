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

    max_length = args.max_length
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_set = TenseTrainDataset('dataset/train.txt', transform=Word2Index(max_length))
    test_set = TenseTestDataset('dataset/test.txt', transform=Word2Index(max_length))

    train_loader = DataLoader(train_set, batch_size=32, num_workers=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=len(test_set), num_workers=8, shuffle=False)

    encoder = TenseEncoder(input_size=Constant.VOCABULARY_SIZE, hidden_size=hidden_size, num_layers=num_layers)
    decoder = TenseDecoder(output_size=Constant.VOCABULARY_SIZE, hidden_size=hidden_size, num_layers=num_layers)

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=Constant.LR, momentum=0.9)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=Constant.LR, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    Model.train(
        {'encoder': encoder, 'decoder': decoder},
        {'encoder_optimizer': encoder_optimizer, 'decoder_optimizer': decoder_optimizer},
        criterion,
        args.epochs,
        train_loader,
        args.annealing,
        args.path
    )

    Model.test(
        'weights/encoder_100.weight',
        'weights/decoder_100.weight',
        test_loader
    )
