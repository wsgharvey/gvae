import math
import torch
import torch.nn as nn

import pygcn
import ptutils as ptu

def test_generator(n, p):
    """
    sample graphs of fixed size as follows:
    - divide nodes into groups 1 and 2
    - sample three types of relations:
      - one exists only between pairs of type 1 nodes
      - one exists only between pairs of type 2 nodes
      - one type connects type 1 and type 2 nodes
    """
    type_1 = torch.rand(n, 1) > 0.5
    type_2 = ~type_1

    # adjacency matrix in case allowed edges existed
    A_allowed = torch.stack([
        type_1 * type_1.t(),
        type_1 * type_2.t(),
        type_2 * type_2.t(),
    ])
    A_allowed = A_allowed * (1-torch.eye(n, n))
    return torch.distributions.Bernoulli(A_allowed*p).sample()

def normalise(t, dim):
    return t / t.norm(2, dim=dim, keepdim=True)

def get_degree(Z):
    Z_sum = Z.sum(dim=0, keepdim=True)
    D = (Z*Z_sum).sum(dim=1)
    return D

def prepare_Z(Z):
    n, k = Z.shape
    Z = normalise(Z, dim=1)
    ones = normalise(torch.ones(n, k), dim=1)
    degree = (get_degree(Z) + get_degree(ones)).view(n, 1)
    mult = degree**(-0.5)
    Z = Z * mult
    ones = ones * mult
    return Z, ones

class GraphiteLayer(nn.Module):

    def __init__(self, in_dim, out_dim, activation):

        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.act = activation

    def forward(self, Z, H):
        """
        Z: n x . tensor
        H: n x in_dim tensor

        A = ZZ'/sqr(norm(Z)) + 11'
        Z* = GNN(A, [Z|X])
        """
        HW = self.linear(H)
        Z_, ones = prepare_Z(Z)
        update_from_Z = Z_ @ (Z_.t() @ HW)
        update_from_1 = ones @ (ones.t() @ HW)
        return self.act(update_from_Z + update_from_1)

class GraphiteBlock(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):

        super().__init__()
        self.l1 = GraphiteLayer(in_dim, hidden_dim, torch.relu)
        self.l2 = GraphiteLayer(hidden_dim, out_dim, torch.relu)

    def forward(self, Z):

        H = self.l1(Z, Z)
        return self.l2(Z, H)


class GraphitePaperDecoder(nn.Module):

    def __init__(self, latent_dim, hidden_dim, final_dim):
        super().__init__()
        self.block = GraphiteBlock(latent_dim, hidden_dim, final_dim)

    def forward(self, Z):
        self.embed(Z)
        return torch.distributions.Bernoulli(
            torch.sigmoid(self.Z_final @ self.Z_final.t()))

    def embed(self, Z):
        self.Z_final = self.block(Z)

    def get_edge_dist(self, edges):
        """
        edges: 2 x . LongTensor of edges
        """
        heads = self.Z_final.index_select(index=edges[0], dim=0)
        tails = self.Z_final.index_select(index=edges[1], dim=0)
        probs = torch.sigmoid((heads * tails).sum(dim=1))
        return torch.distributions.Bernoulli(probs)


class GraphiteEncoder(nn.Module):
    def __init__(self, n, initial_dim, hidden_dim, latent_dim):
        super().__init__()
        self.init_hidden = nn.Parameter(torch.randn(n, initial_dim))
        self.gc1 = pygcn.layers.GraphConvolution(initial_dim, hidden_dim, bias=False)
        self.gc2 = pygcn.layers.GraphConvolution(hidden_dim, latent_dim*2, bias=False)

    def forward(self, A):
        A_hat = A + torch.eye(*A.shape)
        H = self.init_hidden
        H = self.gc1(H, A_hat)
        H = torch.relu(H)
        H = self.gc2(H, A_hat)
        H = H.view(H.shape[0], 2, -1)
        means = H[:, 0]
        log_stddevs = H[:, 1]
        stddevs = log_stddevs.exp()
        return torch.distributions.Normal(means, stddevs)

class GraphiteVAE(ptu.Trainable, ptu.CudaCompatibleMixin, ptu.HasDataloaderMixin):
    p_Z = torch.distributions.Normal(0, 1)

    def init_nn(self, n, latent_dim, encoder_hidden_dim, decoder_hidden_dim):
        self.n = n
        self.latent_dim = latent_dim
        # init neural nets
        self.encoder = GraphiteEncoder(
            n, initial_dim=encoder_hidden_dim,
            hidden_dim=encoder_hidden_dim, latent_dim=latent_dim)
        self.decoder = GraphitePaperDecoder(
            latent_dim=latent_dim, hidden_dim=decoder_hidden_dim,
            final_dim=decoder_hidden_dim)

    def elbo(self, A):
        q_Z = self.encoder(A)
        Z = q_Z.sample()
        p_A_given_Z = self.decoder(Z)
        estimated_elbo = p_A_given_Z.log_prob(A).sum() \
            + self.p_Z.log_prob(Z).sum() \
            - q_Z.log_prob(Z).sum()
        return estimated_elbo

    def sample(self):
        Z = self.p_Z.sample((self.n, self.latent_dim))
        p_A_given_Z = self.decoder(Z)
        return p_A_given_Z.sample()

    def sample_reconstruction_mean(self, A):
        q_Z = self.encoder(A)
        Z = q_Z.sample()
        print(Z)
        p_A_given_Z = self.decoder(Z)
        return p_A_given_Z.probs

    def loss(self, A):
        elbo = self.elbo(A)
        self.log = {'elbo': elbo.item()}
        return -elbo









class MultiRelationGraphiteLayer(nn.Module):

    def __init__(self, r, k0, k1, k2, activation):
        """
        r: number of relations
        k0: incoming hidden dim (of Z)
        k1: incoming hidden dim (of H)
        k2: outgoing hidden dim
        """
        super().__init__()
        self.r = r
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2

        def init_transformation_matrix(param):
            stdv = 1. / math.sqrt(param.size(1))
            param.data.uniform_(-stdv, stdv)

        self.R = nn.Parameter(torch.Tensor(r, k0))
        init_transformation_matrix(self.R)
        self.W = nn.Parameter(torch.Tensor(r, k2, k1))
        init_transformation_matrix(self.W)

        self.relation_embeddings = 0
        self.act = activation

    def forward(self, Z, H):
        """
        Z* = \sum_r (Z @ G_r @ Z') @ H @ W
        """
        # Z : n x k0
        # R : r x k0
        # H : n x k1
        # W : r x k1 x k2

        Z = Z.unsqueeze(0)
        Z = Z / torch.norm(Z, 2, dim=1, keepdim=True)
        ZR = Z*self.R.view(self.r, 1, self.k0)

        def chain_bmm(ms, batch_size):
            # multiply from right
            prod = ms[-1].expand(batch_size, -1, -1)
            for m in ms[:-1:-1]:
                m= m.expand(batch_size, -1, -1)
                prod = torch.bmm(m, prod)
            return prod

        HW = chain_bmm([H.unsqueeze(0), self.W], self.r)
        one_one_t_HW = HW.sum(dim=0, keepdim=True)

        product = chain_bmm([ZR, Z.transpose(1, 2), HW], self.r) + one_one_t_HW

        return self.act(product.sum(dim=0))

n = 20
r = 3
k0 = 6
k1 = 6
k2 = 6

Z = torch.randn(n, k0)
R = torch.randn(r, k0)
H = torch.randn(n, k1)
W = torch.randn(r, k1, k2)

class AttentiveMultiRelationGraphiteLayer(nn.Module):

    def __init__(self, in_dim, out_dim, n_relations, activation):

        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.relation_embeddings = ...
        self.act = activation
        self.G_r = nn.Parameter(torch.zeros(n_relations, in_dim))

    def forward(self, Z, H):
        """
        Z* = \sum_r (Z @ G_r @ Z') @ H @ W
        """
        HW = self.linear(H).view(self.n, self.n_relations, -1)

        # (n x .) x (. x n)
        Z @ Z.t() @ self.G_r

        # (r x n x n) x (n x r x .) -> n x .
        A @ HW
