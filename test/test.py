import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

is_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')

standardiazator = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))
])

train_data = dsets.MNIST(root='./data', train=True, transform=standardiazator, download=True)
test_data = dsets.MNIST(root='./data', train=False, transform=standardiazator, download=True)

batch_size = 200
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True)

import numpy as np
import matplotlib.pyplot as plt

def imshow(img):
    img = (img + 1) / 2
    img = img.squeeze()
    np_img = img.numpy()
    plt.imshow(np_img, cmap='gray')
    plt.savefig('out1.png')
    
def imshow_grid(img, name='out'):
    img = utils.make_grid(img.cpu().detach())
    img = (img + 1) / 2
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(f'{name}.png')
    
d_noise = 100
d_hidden = 256

def sample_z(batch_size=1, d_noise=100):
    return torch.randn(batch_size, d_noise, device=device)

G = nn.Sequential(
    nn.Linear(d_noise, d_hidden),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(d_hidden, d_hidden),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(d_hidden, 28*28),
    nn.Tanh()
).to(device)

# example_mini_batch_img, example_mini_batch_label = next(iter(train_data_loader))
# imshow_grid(example_mini_batch_img[:16, :, :, :])

z = sample_z()
img_fake = G(z).view(-1, 28, 28)
imshow(img_fake.squeeze().cpu().detach())

z = sample_z(16)
img_fake = G(z)
imshow_grid(img_fake)

D = nn.Sequential(
    nn.Linear(28*28, d_hidden),
    nn.LeakyReLU(),
    nn.Dropout(0.1),
    nn.Linear(d_hidden, d_hidden),
    nn.LeakyReLU(),
    nn.Dropout(0.1),
    nn.Linear(d_hidden, 1),
    nn.Sigmoid()
).to(device)

print(G(z).shape)
print(D(G(z)).shape)
print(D(G(z))[0:5].transpose(0, 1), D(G(z))[0:5].transpose(0, 1).shape)

# Train

criterion = nn.BCELoss()

def run_epoch(generator, discriminator, _optimizer_g, _optimizer_d):
    generator.train()
    discriminator.train()
    
    for img_batch, label_batch in train_data_loader:
        img_batch, label_batch = img_batch.to(device), label_batch.to(device)
        
        _optimizer_d.zero_grad()
        
        p_real = discriminator(img_batch.view(-1, 28*28))
        p_fake = discriminator(generator(sample_z(batch_size, d_noise)))
        
        loss_real = -1 * torch.log(p_real)
        loss_fake = -1 * torch.log(1. - p_fake).mean()
        loss_d = (loss_real + loss_fake).mean()
        
        loss_d.backward()
        _optimizer_d.step()
        
        _optimizer_g.zero_grad()
        p_fake = discriminator(generator(sample_z(batch_size, d_noise)))
        
        loss_g = -1 * torch.log(p_fake).mean()
        loss_g.backward()
        _optimizer_g.step()
        
def evaluate_model(generator, discriminator):
    p_real, p_fake = 0., 0.
    
    generator.eval()
    discriminator.eval()
    
    for img_batch, label_batch in test_data_loader:
        img_batch, label_batch = img_batch.to(device), label_batch.to(device)
        with torch.autograd.no_grad():
            p_real += (torch.sum(discriminator(img_batch.view(-1, 28*28))).item()) / 10000.
            p_fake += (torch.sum(discriminator(generator(sample_z(batch_size, d_noise)))).item()) / 10000.

    return p_real, p_fake

def init_params(model):
    for p in model.parameters():
        if(p.dim() > 1):
            nn.init.xavier_normal_(p)
        else:
            nn.init.uniform_(p, 0.1, 0.2)

init_params(G)
init_params(D)

optimizer_g = optim.Adam(G.parameters(), lr=0.0002)
optimizer_d = optim.Adam(D.parameters(), lr=0.0002)

p_real_trace = []
p_fake_trace = []

for epoch in range(200):
    run_epoch(G, D, optimizer_g, optimizer_d)
    p_real, p_fake = evaluate_model(G, D)
    
    p_real_trace.append(p_real)
    p_fake_trace.append(p_fake)
    
    if((epoch + 1) % 50 == 0):
        print("Epoch: {}, P(real): {:.4f}, P(fake): {:.4f}".format(epoch + 1, p_real, p_fake))
        imshow_grid(G(sample_z(16)).view(16, 1, 28, 28), name=str(epoch))