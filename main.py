import argparse

import numpy as np
import torch

from gan import WGAN_GP
from figure import Figure


def sample_from_data(data, batch_size):
    return torch.from_numpy(data[np.random.randint(len(data), size=(batch_size, ))]).float()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", type=str, required=True)
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    if args.dist == "normal":
        data = np.random.normal(loc=3, scale=0.5, size=(10000, 1))
    elif args.dist == "uniform":
        data = np.random.uniform(1, 5, size=(10000, 1))
    elif args.dist == "bimodal":
        data1 = np.random.normal(loc=3, scale=0.5, size=(3333, 1))
        data2 = np.random.normal(loc=2, scale=0.2, size=(3333, 1))
        data3 = np.random.normal(loc=4, scale=0.1, size=(3333, 1))
        data = np.vstack([data1, data2, data3])
    else:
        raise NotImplementedError

    batch_size = 256
    n_critics = 5

    gan = WGAN_GP(G_input_dim=4, G_output_shape=(1, ), use_custom_gp=True)
    fig = Figure(save_animation=True, save_name=f"gifs/{args.dist}")

    ws = []
    for i in range(3000):

        print("Step:", i+1)

        for j in range(n_critics):

            w = gan.update_D_one_iter(real_data_batch=sample_from_data(data, batch_size))
            if j == n_critics - 1:
                ws.append(w)

        gan.update_G_one_iter(batch_size)

        if (i+1) % 100 == 0:
            fig.replot(ws=ws, critic=gan.D, fake_data_batch=gan.sample_G(batch_size=1000), real_data_all=data)

    fig.finalize()
