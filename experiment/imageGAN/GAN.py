import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, d_noise, d_hidden, device='cpu'):
        super(Generator, self).__init__()
        self.G = nn.Sequential(
            nn.Linear(d_noise, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, 28*28),
            nn.Tanh()
        ).to(device)

        self._init_params(self.G)

    def _init_params(self, model):
        for p in model.parameters():
            if(p.dim() > 1):
                nn.init.xavier_normal_(p)
            else:
                nn.init.uniform_(p, 0.1, 0.2)
    
    def forward(self, x):
        return self.G(x)

class Discriminator(nn.Module):
    def __init__(self, d_hidden, device='cpu'):
        super(Discriminator, self).__init__()
        self.D = nn.Sequential(
            nn.Linear(28*28, d_hidden),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid()
        ).to(device)

        self._init_params(self.D)
    
    def _init_params(self, model):
        for p in model.parameters():
            if(p.dim() > 1):
                nn.init.xavier_normal_(p)
            else:
                nn.init.uniform_(p, 0.1, 0.2)

    def forward(self, x):
        return self.D(x)

def sample_z(batch_size=1, d_noise=100, device='cpu'):
    return torch.randn(batch_size, d_noise, device=device)