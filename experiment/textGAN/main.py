import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn

import torch.optim as optim
import random

from Class.Dataset import Dataset
from Class.GAN import TextGAN

import wandb

# LSTM 설정
input_size = 10  # 입력 크기
hidden_size = 5  # 은닉 상태 크기
num_layers = 1  # LSTM 레이어 수



def run_epoch(generator, discriminator, g_optimizer, d_optimizer, d_criterion, dataset, logger):
    # generator.train()
    # discriminator.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    k = 5
    
    loss_real = 0
    loss_fake = 0
    loss_d = 0
    
    for text_batch in train_data_loader:
        sentence = text_batch[0]
        # print('tb :', text_batch)
        # sentence = sentence.to(device)
        sentence_length = len(sentence.split())
        # Train Discriminator
        d_optimizer.zero_grad()
        
        p_real = discriminator(sentence)
        
        random_indices = random.sample(range(1, sentence_length), k)
        mask_array = [True if i in random_indices else False for i in range(sentence_length)]
        
        p_fake = discriminator(generator(sentence, mask_array))
        
        # print('ppo ', p_real, p_fake)/
        
        # print("=====================================")
        # print("p_real : ", p_real)
        # print("p_fake : ", p_fake)
        
        loss_real = -1 * torch.log(p_real)
        loss_fake = -1 * torch.log(1. - p_fake + 1e-9).mean()
        loss_d = (loss_real + loss_fake).mean()
        
        # print('real : ', loss_real, loss_real.shape)
        # print('fake : ', loss_fake, loss_fake.shape)
        # print('d : ', loss_d, loss_d.shape)
        
        
        loss_d.backward()
        d_optimizer.step()
        
        # Train Generator
        g_optimizer.zero_grad()
        p_fake = discriminator(generator(sentence, mask_array))
        
        logger.log({"p_fake": p_fake})
        loss_g = -1 * torch.log(p_fake).mean()
        loss_g.backward()
        g_optimizer.step()
        
    logger.log({"loss_real": loss_real, "loss_fake": loss_fake, "loss_d": loss_d})

def predict(dataset, model, text, next_words=100):
    model.eval()
    
    words = text.split()
    state_h, state_c = model.init_state(len(words))
    
    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        # print(x)
        # print("x shape :",  x.shape)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        
        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])
    return words


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    
    args = {
        "learning_rate": 0.001,
        "epochs": 1000,
        "batch_size": 1
    }
    logger = wandb.init(project="text-gan", mode="online")
    logger.config.update(args)
    
    dataset = Dataset()
    
    textgan = TextGAN(dataset)
    
    g_optimizer = optim.Adam(textgan.generator.parameters(), lr=args['learning_rate'])
    d_optimizer = optim.Adam(textgan.discriminator.parameters(), lr=args['learning_rate'])
    
    d_criterion = nn.BCEWithLogitsLoss()
     
    for epoch in tqdm(range(args['epochs'])):
        # print('\nEpoch :', epoch)
        run_epoch(textgan.generator, textgan.discriminator, 
              g_optimizer, d_optimizer, d_criterion, dataset, logger)
        
    # save model
    torch.save(textgan.generator.state_dict(), 'generator.pth')
    torch.save(textgan.discriminator.state_dict(), 'discriminator.pth')
    # generator.predict('censored censored')
    
    # output_text = predict(dataset, model, text='think think, is this?')
    # print(output_text)