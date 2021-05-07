import logging
import argparse

import Constant

logger = logging.getLogger('__main__.utils')


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-H', '--hidden_size', type=int, default=256)
    parser.add_argument('-a', '--annealing', type=str, default='cyclical', choices=['monotonic', 'cyclical'])
    parser.add_argument('-s', '--save_path', type=str, default='weights')
    parser.add_argument('-t', '--trainable', action='store_true')
    parser.add_argument('-l', '--load', type=str, nargs='+', default=None)
    parser.add_argument('-p', '--period', type=int, nargs='+', default=[1, 1, 1])

    args = parser.parse_args()

    print('=' * 50)

    for key, value in vars(args).items():
        logger.info(f'{key}: {value}')

    logger.info(f'PAD_TOKEN: {Constant.PAD_TOKEN}')
    logger.info(f'SOS_TOKEN: {Constant.SOS_TOKEN}')
    logger.info(f'EOS_TOKEN: {Constant.EOS_TOKEN}')
    logger.info(f'ALP_TOKEN: {Constant.ALP_TOKEN}')

    logger.info(f'LR: {Constant.LR}')
    logger.info(f'NUM_LAYERS: {Constant.NUM_LAYERS}')
    logger.info(f'MAX_LENGTH: {Constant.MAX_LENGTH}')
    logger.info(f'LATENT_SIZE: {Constant.LATENT_SIZE}')
    logger.info(f'VOCABULARY_SIZE: {Constant.VOCABULARY_SIZE}')
    logger.info(f'CONDITION_CATEGORY: {Constant.CONDITION_CATEGORY}')
    logger.info(f'CONDITION_EMBEDDING_SIZE: {Constant.CONDITION_EMBEDDING_SIZE}')
    logger.info(f'BIDIRECTIONAL: {Constant.BIDIRECTIONAL}')

    print('=' * 50)

    return args
