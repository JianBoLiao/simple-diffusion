from PIL import Image
import numpy as np
import imageio

img_dir = './imgs/'
origin_dir ='./imgs/origin/'
pixel_shuffle_dir ='./imgs/pixelshuffle/'
synergy_dysample_dir ='./imgs/synergy_dysample/'
combine_dysample_dir ='./imgs/combine_dysample/'
synergy_pixelshuffle_dir ='./imgs/synergy_pixelshuffle/'
synergy_dysample_relu_dir ='./imgs/synergy_dysample_relu/'
def process(dirs, epoch_num):
    dir_num = len(dirs)
    total_img = Image.new('RGB', ((16 + 2) * 8 - 2, (16 + 2) * dir_num - 2))
    images = []
    for j in range(epoch_num):
        for i, dir in enumerate(dirs):
            img = Image.open(dir + str(j) + '.png')
            total_img.paste(img, (0, i * (16 + 2)))
        total_img.save(img_dir + f'total{j}.png')

        images.append(np.array(total_img))
    
    imageio.mimsave(img_dir + 'total.gif', images, duration=0.2)
if __name__ == '__main__':
    dirs = [origin_dir, pixel_shuffle_dir, synergy_pixelshuffle_dir, synergy_dysample_dir, combine_dysample_dir]
    shuffle_dir = [pixel_shuffle_dir, synergy_pixelshuffle_dir]
    synergy_dir = [synergy_dysample_dir, synergy_dysample_relu_dir]
    process(dirs, 32)    