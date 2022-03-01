import torch

from models import *
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils_ming import SimpleProgressBar

net_embed = ResNet34_embed(64)
checkpoint = torch.load('/home/ylab3/improved_CcGAN/365_data/'
                        'RC-49_256x256/CcGAN-improved/output/embed_models/'
                        'ckpt_ResNet34_embed_epoch_200_seed_2021.pth')
keys = list(checkpoint['net_state_dict'].keys())
for i in range(len(keys)) :
    temp = keys[i].replace('module.','')
    checkpoint['net_state_dict'][temp] = checkpoint['net_state_dict'].pop(keys[i])
net_embed.load_state_dict(checkpoint['net_state_dict'])
net_embed = net_embed.cuda()

net_y2h = model_y2h(64)
checkpoint_y2h = torch.load('/home/ylab3/improved_CcGAN/365_data/'
                            'RC-49_256x256/CcGAN-improved/output/embed_models/'
                            'ckpt_net_y2h_epoch_500_seed_2021.pth')
keys = list(checkpoint_y2h['net_state_dict'].keys())
for i in range(len(keys)) :
    temp = keys[i].replace('module.','')
    checkpoint_y2h['net_state_dict'][temp] = checkpoint_y2h['net_state_dict'].pop(keys[i])
net_y2h.load_state_dict(checkpoint_y2h['net_state_dict'])
net_y2h = net_y2h.cuda()
check_path = '/home/ylab3/improved_CcGAN/365_data/RC-49_256x256/CcGAN-improved/output/output_CcGAN_arch_SAGAN/' \
             'saved_models/CcGAN_SAGAN_soft_nDsteps_2_checkpoint_intrain'

name = 'CcGAN_checkpoint_niters_{}.pth'.format(22500)
gener_path = os.path.join(check_path, name)
net_generator = CcGAN_SAGAN_Generator(dim_z=256, dim_embed=64)
check_ge = torch.load(gener_path)

keys = list(check_ge['netG_state_dict'].keys())
for i in range(len(keys)) :
    temp = keys[i].replace('module.','')
    check_ge['netG_state_dict'][temp] = check_ge['netG_state_dict'].pop(keys[i])

net_generator.load_state_dict(check_ge['netG_state_dict'])
net_generator.cuda()


# def fn_sampleGAN_given_labels(labels, batch_size):
#     fake_images, fake_labels = sample_ccgan_given_labels(net_generator, net_y2h, labels, batch_size=batch_size, to_numpy=True,
#                                                          denorm=True, verbose=True)
#     return fake_images, fake_labels

eval_labels = np.linspace(16.3176, 39.7025)
eval_labels_norm = eval_labels/39.7025
fake_labels = eval_labels_norm

nfake = len(eval_labels_norm)
batch_size = nfake
fake_labels = np.concatenate((eval_labels_norm, eval_labels_norm[0:batch_size]))


fake_images = []

net_generator.eval()
net_y2h.eval()

with torch.no_grad() :
    pd = SimpleProgressBar()
    n_img_got = 0
    while n_img_got < nfake :
        z = torch.randn(batch_size, 256, dtype=torch.float).cuda()
        y = torch.from_numpy(fake_labels[n_img_got:(n_img_got+batch_size)]).type(torch.float).view(-1,1).cuda()

        batch_fake_images = net_generator(z,net_y2h(y))

        assert batch_fake_images.max().item()<=1.0 and batch_fake_images.min().item() >=-1.0
        batch_fake_images = batch_fake_images*0.5 + 0.5
        batch_fake_images = batch_fake_images*255.0
        batch_fake_images = batch_fake_images.type(torch.uint8)

        fake_images.append(batch_fake_images.cpu())
        n_img_got += batch_size
        pd.update(min(float(n_img_got)/nfake, 1) * 100)


fake_images = torch.cat(fake_images,dim = 0)
fake_images = fake_images[0:nfake]
fake_labels = fake_labels[0:nfake]

fake_images = fake_images.numpy()

dump_fake_images_folder = 'z_same_test' + '/fake_images_for_NIQE_nfake{}'.format(len(fake_images))
os.makedirs(dump_fake_images_folder, exist_ok=True)

for k in tqdm(range(len(fake_images))):
    label_i = fake_labels[k] * 39.7025
    filename_i = dump_fake_images_folder + "/{}_{}.png".format(k, label_i)
    os.makedirs(os.path.dirname(filename_i), exist_ok=True)
    image_i = fake_images[k].astype(np.uint8)
    # image_i = ((image_i*0.5+0.5)*255.0).astype(np.uint8)
    image_i_pil = Image.fromarray(image_i.transpose(1, 2, 0))
    image_i_pil.save(filename_i)

