import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as utils


def imsave(img):
    img = (img + 1) / 2
    img = img.squeeze()
    np_img = img.numpy()
    plt.imshow(np_img, cmap='gray')
    plt.savefig('out1.png')
    
def imsave_grid(img, name='out'):
    img = utils.make_grid(img.cpu().detach())
    img = (img + 1) / 2
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig(f'{name}.png')
    