import argparse


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-s', '--size', type=int, dest='hidden_size', default=256)
    parser.add_argument('-n', '--num_layers', type=int, default=2)
    parser.add_argument('-a', '--annealing', type=str, default='cyclical', choices=['monotonic', 'cyclical'])
    parser.add_argument('-p', '--path', type=str, default='weights')
    parser.add_argument('-t', '--trainable', action='store_true')

    args = parser.parse_args()

    print('=' * 50)
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('=' * 50)

    return args
