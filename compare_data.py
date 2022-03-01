import torch
from models import *
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from utils import IMGs_dataset

net_embed = ResNet34_regre_eval(1)
checkpoint = torch.load('/home/ylab3/improved_CcGAN/365_data/RC-49_256x256/'
                        'CcGAN-improved/output/eval_models/'
                        'ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2021_CVMode_False.pth')
keys = list(checkpoint['net_state_dict'].keys())
for i in range(len(keys)) :
    temp = keys[i].replace('module.','')
    checkpoint['net_state_dict'][temp] = checkpoint['net_state_dict'].pop(keys[i])
net_embed.load_state_dict(checkpoint['net_state_dict'])
net_embed= net_embed.cuda()
net_embed.eval()

list_dir = []
for i in range(12) :
    list_dir.append('fake_images_for_NIQE_nfake{}_5000'.format(i))

min_value = 16.3176
max_value = 39.7025

mse_loss = nn.MSELoss()
mse_loss = mse_loss.cuda()
real_loss = []
for i in tqdm(range(len(list_dir))) :
    label_all = []
    image_all = []
    root_path = os.path.join('compared', list_dir[i])
    image_dir_name = os.listdir(root_path)
    for j in range(len(image_dir_name)) :
        label_all.append(float(image_dir_name[j].split('_')[1][:-4]))
        image_array=np.array(Image.open(os.path.join(root_path, image_dir_name[j])))
        image_array = np.transpose(image_array,(2,0,1))
        image_all.append(image_array)

    image_all = np.array(image_all)
    label_all = np.array(label_all)
    label_all = label_all.reshape(5000,1)
    label_all_norm = label_all/39.7025

    trainset = IMGs_dataset(image_all, label_all_norm, normalize=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True,
                                              num_workers=0)

    loss_list = []

    for k,(image_, label_) in enumerate(trainloader) :
        image_=image_.cuda()
        image_ = image_.type(torch.float).cuda()
        label_=label_.cuda()
        label_ = label_.type(torch.float).cuda()
        prediction = net_embed(image_)[0]
        loss = mse_loss(label_,prediction)
        loss_list.append(loss.item())

    real_loss.append(np.mean(loss_list))

print(real_loss)





