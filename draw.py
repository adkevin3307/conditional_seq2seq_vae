import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_log(path: str) -> list:
    history = []

    with open(path, 'r') as log_file:
        index = 0

        for line in log_file:
            if 'Epoch' in line:
                index = 1
                history.append({})

                continue

            if index > 0:
                index += 1

                message = line.split('] ')[-1]
                tokens = message.split(', ')

                for token in tokens:
                    key, value = token.split(': ')[0], token.split(': ')[1]

                    history[-1][key] = float(value)

            if index == 4:
                index = 0

    return history


def show_log(history: list, segment: int) -> None:
    kld_loss = list(map(lambda x: x['kld_loss'], history))
    ce_loss = list(map(lambda x: x['ce_loss'], history))
    bleu4 = list(map(lambda x: x['bleu4'], history))
    test_bleu4 = list(map(lambda x: x['test_bleu4'], history))
    kld_alpha = list(map(lambda x: x['kld_alpha'], history))
    tf_rate = list(map(lambda x: x['tf_rate'], history))
    gaussian = list(map(lambda x: x['gaussian'], history))

    kld_loss = np.mean(np.array(kld_loss).reshape(-1, segment), axis=1).tolist()
    ce_loss = np.mean(np.array(ce_loss).reshape(-1, segment), axis=1).tolist()
    bleu4 = np.mean(np.array(bleu4).reshape(-1, segment), axis=1).tolist()
    test_bleu4 = np.mean(np.array(test_bleu4).reshape(-1, segment), axis=1).tolist()
    kld_alpha = np.mean(np.array(kld_alpha).reshape(-1, segment), axis=1).tolist()
    tf_rate = np.mean(np.array(tf_rate).reshape(-1, segment), axis=1).tolist()
    gaussian = np.mean(np.array(gaussian).reshape(-1, segment), axis=1).tolist()

    lines = []
    figure, axes = plt.subplots(figsize=(15, 8))

    lines += axes.plot(kld_loss, label='kld_loss', color='tab:blue')
    lines += axes.plot(ce_loss, label='ce_loss', color='tab:orange')
    axes.set_ylabel('Loss')

    axes = axes.twinx()

    lines += axes.plot(bleu4, '.', label='bleu4', color='tab:green')
    lines += axes.plot(test_bleu4, '.', label='test_bleu4', color='tab:red')
    lines += axes.plot(kld_alpha, '--', label='kld_alpha', color='tab:purple')
    lines += axes.plot(tf_rate, '--', label='tf_rate', color='tab:brown')
    lines += axes.plot(gaussian, '.', label='gaussian', color='tab:pink')
    axes.set_ylabel('Score / Weight')

    axes.legend(lines, [line.get_label() for line in lines])

    figure.tight_layout()
    plt.show()

    plt.plot(kld_loss, label='kld_loss')
    plt.plot(ce_loss, label='ce_loss')
    plt.legend()
    plt.show()

    plt.plot(kld_alpha, label='kld_alpha')
    plt.plot(tf_rate, label='tf_rate')
    plt.legend()
    plt.show()

    plt.plot(bleu4, label='bleu4')
    plt.plot(test_bleu4, label='test_bleu4')
    plt.plot(gaussian, label='gaussian')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True)
    parser.add_argument('-s', '--segment', type=int, default=1)
    args = parser.parse_args()

    history = parse_log(args.path)
    print(f'history: {len(history)}')

    show_log(history, args.segment)
