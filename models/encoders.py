import torch.nn as nn

import layers.hyp_layers as hyp_layers
import manifolds


class Encoder(nn.Module):
    def __init__(self, c):
        super(Encoder, self).__init__()
        self.c = c

    def encode(self, x, adj):
        if self.encode_graph:
            input = (x, adj)
            output, _ = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output


class HGCF(Encoder):
    def __init__(self, c, args):
        super(HGCF, self).__init__(c)
        self.manifold = getattr(manifolds, "Hyperboloid")()
        assert args.num_layers > 1

        hgc_layers = []
        in_dim = out_dim = args.embedding_dim
        hgc_layers.append(
            hyp_layers.HyperbolicGraphConvolution(
                self.manifold, in_dim, out_dim, self.c, args.network, args.num_layers
            )
        )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encode(self, x, adj):
        x_hyp = self.manifold.proj(x, c=self.c)
        return super(HGCF, self).encode(x_hyp, adj)
