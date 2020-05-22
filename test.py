import numpy as np
import torch
import ptutils as ptu

# from model import GraphiteVAE


class TestGraphDataset(torch.utils.data.Dataset):

    def __init__(self, n, indices):
        self.n = n
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):

        # shuffled ordering of elements
        nodes = np.arange(self.n)
        with ptu.RNG(self.indices[i]):
            np.random.shuffle(nodes)

        A = torch.zeros(self.n, self.n)
        pre = nodes[0]
        for nex in nodes[1:]:
            A[nex, pre] = 1
            A[pre, nex] = 1
            pre = nex

        return A

class TestGraphLoader(TestGraphDataset):
    def __iter__(self):
        indices = np.arange(len(self))
        np.random.shuffle(indices)
        for i in indices:
            yield self[i]

class TestGraphiteVAE(GraphiteVAE):
    n = 20

    @staticmethod
    def easy_init():
        Class = TestGraphiteVAE
        return Class(
            seed=1,
            nn_args=dict(n=TestGraphiteVAE.n,
                         latent_dim=10,
                         encoder_hidden_dim=12,
                         decoder_hidden_dim=14),
            optim_args=dict(type=torch.optim.Adam,
                            lr=1e-6))

    def post_init(self):
        self.set_dataloaders(
            TestGraphLoader(self.n, list(range(0, 200))),
            TestGraphLoader(self.n, list(range(0, 50))),
            TestGraphLoader(self.n, list(range(0, 50))),
        )
        self.set_save_valid_conditions('valid', 'every', 1, 'epochs')

gvae = TestGraphiteVAE.easy_init()
gvae.train_n_epochs(500)
