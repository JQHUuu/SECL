import torch
import torch.nn as nn
import torch.nn.functional as F


class my_model(nn.Module):
    def __init__(self, dims, num_nodes, k, name):
        super(my_model, self).__init__()
        if name == "citeseer" or name == "acm" or name == "amap" or name == "bat":
            self.layers1 = nn.Linear(dims[0], dims[1])
            self.layers2 = nn.Linear(num_nodes, 1024)
            self.layers3 = nn.Linear(dims[1], k)
            self.layers4 = nn.Linear(1024, dims[1])
        else:
            self.layers1 = nn.Linear(dims[0], dims[1])
            self.layers2 = nn.Linear(num_nodes, dims[1])
            self.layers3 = nn.Linear(dims[1], k)

    def forward(self, x, A, name, is_train=True, sigma=0.01):
        if name == "citeseer" or name == "acm" or name == "amap" or name == "bat":
            out1 = self.layers1(x)
            out2 = F.relu(self.layers2(A))
            out2 = F.normalize(out2, dim=1, p=2)
            out2 = self.layers4(out2)

            z1 = F.normalize(F.relu(out1))
            z2 = F.normalize(F.relu(out2))
            out1 = F.normalize(out1, dim=1, p=2)
            out2 = F.normalize(out2, dim=1, p=2)

            out3 = F.softmax(self.layers3(z1), dim=1)
        else:
            out1 = self.layers1(x)
            out2 = self.layers2(A)

            z1 = F.normalize(F.relu(out1))
            z2 = F.normalize(F.relu(out2))
            out1 = F.normalize(out1, dim=1, p=2)
            out2 = F.normalize(out2, dim=1, p=2)

            out3 = F.softmax(self.layers3(z1), dim=1)
        return out1, out2, out3


class mu(nn.Module):
    def __init__(self, dims):
        super(mu, self).__init__()
        self.layers1 = nn.Linear(dims, 7)

    def forward(self, z, is_train=True, sigma=0.01):

        out1 = F.sigmoid(self.layers1(z))
        out1 = F.normalize(out1, dim=1, p=2)
        return out1