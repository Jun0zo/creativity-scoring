import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import GAN
import utils




def run_epoch(generator, discriminator, _optimizer_g, _optimizer_d, batch_size, d_noise, device):
    generator.train()
    discriminator.train()
    
    for img_batch, label_batch in train_data_loader:
        img_batch, label_batch = img_batch.to(device), label_batch.to(device)
        
        _optimizer_d.zero_grad()
        
        p_real = discriminator(img_batch.view(-1, 28*28))
        p_fake = discriminator(generator(GAN.sample_z(batch_size, d_noise, device)))
        
        loss_real = -1 * torch.log(p_real)
        loss_fake = -1 * torch.log(1. - p_fake).mean()
        loss_d = (loss_real + loss_fake).mean()
        
        loss_d.backward()
        _optimizer_d.step()
        
        _optimizer_g.zero_grad()
        p_fake = discriminator(generator(GAN.sample_z(batch_size, d_noise, device)))
        
        loss_g = -1 * torch.log(p_fake).mean()
        loss_g.backward()
        _optimizer_g.step()
        
def evaluate_model(generator, discriminator, batch_size, d_noise, device):
    p_real, p_fake = 0., 0.
    
    generator.eval()
    discriminator.eval()
    
    for img_batch, label_batch in test_data_loader:
        img_batch, label_batch = img_batch.to(device), label_batch.to(device)
        with torch.autograd.no_grad():
            p_real += (torch.sum(discriminator(img_batch.view(-1, 28*28))).item()) / 10000.
            p_fake += (torch.sum(discriminator(generator(GAN.sample_z(batch_size, d_noise, device)))).item()) / 10000.

    return p_real, p_fake


if __name__ == "__main__":
    # cuda device settings
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')

    # dataset
    standardiazator = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5))
    ])

    train_data = dsets.MNIST(root='./data', train=True, transform=standardiazator, download=True)
    test_data = dsets.MNIST(root='./data', train=False, transform=standardiazator, download=True)


    batch_size = 200
    d_noise = 100
    d_hidden = 256

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True) # 

    G = GAN.Generator(d_noise=100, d_hidden=256, device=device)
    D = GAN.Discriminator(d_hidden=256, device=device)
    criterion = nn.BCELoss()

    z = GAN.sample_z(device=device)
    img_fake = G(z).view(-1, 28, 28)
    utils.imsave(img_fake.squeeze().cpu().detach())

    z = GAN.sample_z(16, device=device)
    img_fake = G(z)
    utils.imsave_grid(img_fake)
    
    print(G(z).shape)
    print(D(G(z)).shape)
    print(D(G(z))[0:5].transpose(0, 1), D(G(z))[0:5].transpose(0, 1).shape)

    optimizer_g = optim.Adam(G.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(D.parameters(), lr=0.0002)

    p_real_trace = []
    p_fake_trace = []

    # training
    for epoch in range(500):
        run_epoch(G, D, optimizer_g, optimizer_d, batch_size, d_noise, device)
        p_real, p_fake = evaluate_model(G, D, batch_size, d_noise, device)
        
        p_real_trace.append(p_real) 
        p_fake_trace.append(p_fake)
        
        if((epoch + 1) % 50 == 0):
            print("Epoch: {}, P(real): {:.4f}, P(fake): {:.4f}".format(epoch + 1, p_real, p_fake))
            utils.imsave_grid(G(GAN.sample_z(16, device=device)).view(16, 1, 28, 28), name=str(epoch))