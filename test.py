from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import train
from tqdm import tqdm
from ContextUnet import *
from diffusion_utilities import *
import pdb
from wpj import load_model
from PIL import Image
import copy
timesteps = 500
beta1 =  1e-4
beta2 = 0.02

height = 16
save_dir1 = './weight/origin/'
img_dir1 = './imgs/origin/'
save_dir2 = './weight/pixelshuffle/'
img_dir2 = './imgs/pixelshuffle/'
save_dir3 = './weight/synergy_dysample/'
img_dir3 = './imgs/synergy_dysample/'
save_dir4 = './weight/combine_dysample/'
img_dir4 = './imgs/combine_dysample/'
save_dir5 = './weight/synergy_pixelshuffle/'
img_dir5 = './imgs/synergy_pixelshuffle/'
save_dir6 = './weight/synergy_dysample_relu/'
img_dir6 = './imgs/synergy_dysample_relu/'
device = torch.device('cuda:0' if torch.cuda.is_available() else torch.device('cpu'))

# construct DDPM noise schedule
b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device = device) + beta1
a_t = 1 - b_t
ab_t = torch.cumsum(a_t.log(), dim = 0).exp()
ab_t[0] = 1

def denoise_add_noise(x, t, pred_noise, z=None):
    if z is None:
        z = torch.randn_like(x)
    noise = b_t.sqrt()[t] * z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

@torch.no_grad()
def sample_ddpm(models, n_sample, save_rate=20):
    samples = torch.randn(n_sample, 3, height, height).to(device)
    samples = [copy.deepcopy(samples) for _ in range(6)]
    intermediates = [[], [], [], [], [], []]
    
    for i in tqdm(range(timesteps, 0, -1)):
        # print(f'sampling timestep {i:3d}', end='\r')

        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)
        z = torch.randn_like(samples[0]) if i > 1 else 0
        # pdb.set_trace()
        for j, model in enumerate(models):
            intermediate = intermediates[j]
            eps = model(samples[j], t)
            samples[j] = denoise_add_noise(samples[j], i, eps, z)
            if i % save_rate ==0 or i==timesteps or i<8:
                intermediate.append(samples[j].detach().cpu().numpy())
            if i==timesteps:
                intermediate = np.stack(intermediate)


    intermediates = np.stack(intermediates)
    return samples, intermediates

def save_img(img_dir, imgs):
    imgs = np.moveaxis(imgs, 2, 4)
    imgs = norm_all(imgs, imgs.shape[0], 8)
    for i in range(32):
        total_img = Image.new('RGB', (8 * (16 + 2) - 2, 16))
        for j in range(8):
            img = imgs[i][j]
            img = Image.fromarray((img * 255).astype(np.uint8))
            total_img.paste(img, (j * (16 + 2), 0))
        total_img.save(img_dir + f'{i}.png')


def main():
    model1 = ContextUnet1(in_channels=3, n_feat=train.n_feat, n_cfeat=train.n_cfeat, height=height)
    model2 = ContextUnet2(in_channels=3, n_feat=train.n_feat, n_cfeat=train.n_cfeat, height=height)
    model3 = ContextUnet3(in_channels=3, n_feat=train.n_feat, n_cfeat=train.n_cfeat, height=height)
    model4 = ContextUnet4(in_channels=3, n_feat=train.n_feat, n_cfeat=train.n_cfeat, height=height)
    model5 = ContextUnet5(in_channels=3, n_feat=train.n_feat, n_cfeat=train.n_cfeat, height=height)
    model6 = ContextUnet6(in_channels=3, n_feat=train.n_feat, n_cfeat=train.n_cfeat, height=height)
    # model = nn.DataParallel(model)
    model1.to(device)
    load_model(model1, f'{save_dir1}model_199.pth')
    model1.eval()

    model2.to(device)
    load_model(model2, f'{save_dir2}model_199.pth')
    model2.eval()

    model3.to(device)
    load_model(model3, f'{save_dir3}model_199.pth')
    model3.eval()

    model4.to(device)
    load_model(model4, f'{save_dir4}model_199.pth')
    model4.eval()

    model5.to(device)
    load_model(model5, f'{save_dir5}model_199.pth')
    model5.eval()

    model6.to(device)
    load_model(model6, f'{save_dir6}model_199.pth')
    model6.eval()
    # if not os.path.exists(img_dir):
    #     os.makedirs(img_dir)
    # plt.clf()
    models = [model1, model2, model3, model4, model5, model6]
    samples, intermediate = sample_ddpm(models, 8)

    # animation_ddpm = plot_sample(intermediate_ddpm,32,4,img_dir, "ani_run", None, save=True)
    save_img(img_dir1, intermediate[0])
    save_img(img_dir2, intermediate[1])
    save_img(img_dir3, intermediate[2])
    save_img(img_dir4, intermediate[3])
    save_img(img_dir5, intermediate[4])
    save_img(img_dir6, intermediate[5])

    

if __name__ == '__main__':
    main()
