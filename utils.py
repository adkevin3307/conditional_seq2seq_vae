import argparse


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
        print(f'{key}: {value}')
    print('=' * 50)

    return args
