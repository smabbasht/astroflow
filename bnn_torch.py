import matplotlib.pyplot as plt
import corner
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import torchbnn as bnn

from dataset import Dataset


class BNN():

    def __init__(self, in_features: int, out_features: int, device: str,
                 datapath: str, activation: str = 'relu',
                 test_size: float = 0.3):
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.activation = activation
        self.model = self.create_model()
        self.datapath = datapath
        self.test_size = test_size

    def create_model(self):
        model = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=1,
                            in_features=self.in_features, out_features=100),
            nn.ReLU() if self.activation == 'relu' else nn.Tanh(),
            # bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=100,
            #                 out_features=200),
            # nn.Tanh() if self.activation == 'relu' else nn.ReLU(),
            # bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=100,
            # out_features=100),
            # nn.ReLU() if self.activation == 'relu' else nn.Tanh(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=1, in_features=100,
                            out_features=100),
            nn.ReLU() if self.activation == 'relu' else nn.Tanh(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=1,
                            in_features=100, out_features=self.out_features),
        ).to(self.device)
        print('BNN model has been created')
        return model

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Model has been saved at '{path}'")

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))
        print(f"Model has been loaded from '{path}'")

    def train(self, n_epochs: int, batch_size: int, n_batches: int,
              lr: float = 0.01, kl_weight: float = 0.01, verbose: bool = True,
              plot_prefix: str = 'plot_bnn'):
        mse_loss = nn.MSELoss()
        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for batch in range(n_batches):
            self.model.train()

            dataset = Dataset(self.datapath, self.test_size,
                              batch_size, 0+batch_size*batch)
            x_train, x_test, y_train, y_test = dataset.fetch_data()

            x_train = torch.tensor(
                x_train, dtype=torch.float32, device=self.device).to(self.device)
            y_train = torch.tensor(
                y_train, dtype=torch.float32, device=self.device).to(self.device)
            x_test = torch.tensor(
                x_test, dtype=torch.float32, device=self.device).to(self.device)
            # y_test = torch.tensor(
            #     y_test, dtype=torch.float32, device=self.device).to(self.device)

            y_train = y_train.unsqueeze(1).unsqueeze(
                2).expand(-1, 100, 5000, -1)

            for step in range(n_epochs):
                print(f"Step no: {step}")
                pre = self.model(x_train)
                mse = mse_loss(pre, y_train)
                kl = kl_loss(self.model)
                cost = mse + kl_weight*kl
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                print(f'Epoch no. {batch}.{step}:- MSE : %2.2f, KL : %2.2f' %
                      (mse.item(), kl.item()))

            self.save_model(f'models/model_{batch}.pth')
            self.model.eval()
            predictions = self.model(x_test).detach().cpu().numpy()
            # elif self.device == 'cpu':
            #     predictions = torch.mean(self.model(
            #         x_test), dim=1).detach().numpy()
            self.plot(predictions, y_test, "figures",
                      f"{plot_prefix}_{batch}")

    def infer(self, data_offset: int, n_examples: int):
        torch.cuda.empty_cache()
        dataset = Dataset(self.datapath, 1,
                          n_examples, data_offset)
        _, x_test, _, y_test = dataset.fetch_data()
        print("y_test shape received in model.infer:", y_test.shape)
        x_test = torch.tensor(
            x_test, dtype=torch.float32, device=self.device).to(self.device)
        # y_test = torch.tensor(
        #     y_test, dtype=torch.float32, device=self.device).to(self.device)
        self.model.eval()
        predictions = self.model(x_test).detach().cpu().numpy()
        return predictions, y_test

    def plot(self, predictions: np.ndarray, true: np.ndarray, save_folder: str,
             filename_prefix: str):
        predictions = np.mean(predictions, axis=1)
        print(predictions.dtype)
        labels = ["alpha", "mass_min", "mass_max", "sigma_ecc"]
        idx = predictions.shape[0]

        print("y_test shape:", true.shape)

        if save_folder[-1] != '/':
            save_folder += '/'
        # print("starting the plotting loop")
        for i in tqdm(range(idx), desc='examples', leave='False'):
            path = osp.join(save_folder, f'{filename_prefix}_{i}')
            # print("idx:", idx, "and i:", i)
            samples = predictions[i]

            # print(f"loop no. {i}")

            print(samples.shape)

            figure = corner.corner(
                samples,
                labels=labels,
                truths=true[i],
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_kwargs={"fontsize": 12}
            )
            # print("done defining figure")

            if i == (idx-1):
                figure.savefig(path)
            plt.close(figure)
