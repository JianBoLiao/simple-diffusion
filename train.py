import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ContextUnet import *
from diffusion_utilities import *
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

timesteps = 500
beta1 =  1e-4
beta2 = 0.02

device = torch.device("cuda" if torch.cuda.is_available() else torch.device('cpu'))
n_feat = 64
n_cfeat = 5
height = 16
save_dir = './weight/synergy_dysample_relu/'
log_dir = './log'

batch_size = 100
n_epoch = 200
lrate = 1e-3

# construct DDPM noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device = device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim = 0).exp()
ab_t[0] = 1

def perturb_input(x, t, noise):
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise

def train(model, dataloader, optimizer):
    pbar = tqdm(dataloader, mininterval=2)
    losses = []
    for x, _ in pbar:   # x: images
        optimizer.zero_grad()
        x = x.to(device)

        noise = torch.randn_like(x)
        t = torch.randint(1, timesteps + 1, (x.shape[0], )).to(device)
        x_perb = perturb_input(x, t, noise)

        pred_noise = model(x_perb, t / timesteps)

        loss = F.mse_loss(pred_noise, noise)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.mean(losses)


def main():
    writer = SummaryWriter(log_dir)
    model = ContextUnet6(in_channels=3, n_feat=n_feat, n_cfeat=n_cfeat, height=height)  
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lrate)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=8, gamma=0.5)

    dataset = CustomDataset("./sprites_1788_16x16.npy", "./sprite_labels_nc_1788_16x16.npy", transform, null_context=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    model.train()
    for epoch in range(n_epoch):
        optimizer.param_groups[0]['lr'] = lrate*(1 - epoch / n_epoch)
        loss = train(model, dataloader, optimizer)
        print('Epoch: {} | Loss: {:.4f}'.format(epoch, loss))
        writer.add_scalar('loss/loss6', loss, epoch)
        # schedule.step()
        if epoch % 10 == 0 or epoch == int(n_epoch - 1):
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            torch.save(model.state_dict(), save_dir + f"model_{epoch}.pth")
            print('saved model at ' + save_dir + f"model_{epoch}.pth")

if __name__ == "__main__":
    main()