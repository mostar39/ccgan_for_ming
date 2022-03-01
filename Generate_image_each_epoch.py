import torch
import torch.nn as nn
import pandas as pd
import copy
from models import *
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from train_ccgan import train_ccgan, sample_ccgan_given_labels
from opts import parse_opts
from utils import SimpleProgressBar

num_of_picture_train = 200
num_of_picture_test = 37


train_base_before_root = '/home/ylab3/dataset_365/side_images/before/train/'
train_base_after_root = '/home/ylab3/dataset_365/side_images/after/train/'
test_base_before_root = '/home/ylab3/dataset_365/side_images/before/train/'
test_base_after_root = '/home/ylab3/dataset_365/side_images/after/train/'

train_before_root = [train_base_before_root+str(i+1)+'.jpg' for i in range(num_of_picture_train)]
train_after_root = [train_base_after_root+str(i+1)+'.jpg' for i in range(num_of_picture_train)]
test_before_root = [test_base_before_root+str(i+1)+'.jpg' for i in range(num_of_picture_test)]
test_after_root = [test_base_after_root+str(i+1)+'.jpg' for i in range(num_of_picture_test)]

train_all_picture_root = train_before_root+train_after_root
test_all_picture_root = test_before_root+test_after_root
#===============================side bmi root============================
train_diagnose_data = pd.read_excel('/home/ylab3/dataset_365/diagnose_BMI/train.xlsx')
test_diagnose_data = pd.read_excel('/home/ylab3/dataset_365/diagnose_BMI/test.xlsx')

train_before_data = []
train_after_data = []
test_before_data = []
test_after_data = []

train_diagnose_data = train_diagnose_data.drop(['Unnamed: 0'], axis=1)
test_diagnose_data = test_diagnose_data.drop(['Unnamed: 0'], axis=1)


for i in range(num_of_picture_train) :
    train_before_data.append(train_diagnose_data['Before_BMI'][i])
    train_after_data.append(train_diagnose_data['After_BMI'][i])
train_all_data = train_before_data + train_after_data

for i in range(num_of_picture_test) :
    test_before_data.append(test_diagnose_data['Before_BMI'][i])
    test_after_data.append(test_diagnose_data['After_BMI'][i])
test_all_data = test_before_data + test_after_data

images_root_all = train_all_picture_root.copy()
labels_all = train_all_data.copy()
labels_all = np.array(labels_all)
labels_all = labels_all.astype(float)

images_all = []

for i in range(len(images_root_all)) :
    image_file = np.array(Image.open(images_root_all[i]))
    image_file=np.transpose(image_file,(2,0,1))
    images_all.append(image_file)

images_all = np.array(images_all)

indx_train = np.arange(0,400)
# hf.close()
print("\n 365 dataset shape: {}x{}x{}x{}".format(images_all.shape[0], images_all.shape[1], images_all.shape[2], images_all.shape[3]))

# data split


images_train = images_all[indx_train]
labels_train_raw = labels_all[indx_train]


# only take images with label in (q1, q2)
q1 = 16.3176
q2 = 39.7025
indx = np.where((labels_train_raw>q1)*(labels_train_raw<q2)==True)[0]
labels_train_raw = labels_train_raw[indx]
images_train = images_train[indx]

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

checkpoint_name_list = os.listdir(check_path)
name_list = []
for i in range(len(checkpoint_name_list)) :
    name = 'CcGAN_checkpoint_niters_{}.pth'.format(2500 + 2500*i)
    name_list.append(name)

def fn_sampleGAN_given_labels(labels, batch_size):
    fake_images, fake_labels = sample_ccgan_given_labels(net_generator, net_y2h, labels, batch_size=batch_size, to_numpy=True,
                                                         denorm=True, verbose=True)
    return fake_images, fake_labels

eval_labels = np.linspace(np.min(labels_all), np.max(labels_all))
eval_labels_norm = eval_labels/39.7025

save_images_folder = '/home/ylab3/improved_CcGAN/365_data/RC-49_256x256/' \
                     'CcGAN-improved/compared'

for i in range(len(checkpoint_name_list)) :
    net_generator = CcGAN_SAGAN_Generator(dim_z=256, dim_embed=64)
    check_ge= torch.load(os.path.join(check_path,name_list[i]))
    keys = list(check_ge['netG_state_dict'].keys())
    for j in range(len(keys)):
        temp = keys[j].replace('module.', '')
        check_ge['netG_state_dict'][temp] = check_ge['netG_state_dict'].pop(keys[j])

    net_generator.load_state_dict(check_ge['netG_state_dict'])
    net_generator.cuda()

    curr_label = eval_labels_norm[i]
    for j in range(len(eval_labels)) :
        curr_label = eval_labels_norm[j]
        if j == 0:
            fake_labels_assigned = np.ones(100) * curr_label
        else:
            fake_labels_assigned = np.concatenate((fake_labels_assigned, np.ones(100) * curr_label))

    fake_images, _ = fn_sampleGAN_given_labels(fake_labels_assigned,100)

    print("\n Dumping fake images for NIQE...")
    dump_fake_images_folder = save_images_folder + '/fake_images_for_NIQE_nfake{}_{}'.format(i,len(fake_images))
    os.makedirs(dump_fake_images_folder, exist_ok=True)
    for k in tqdm(range(len(fake_images))):
        label_i = fake_labels_assigned[k] * 39.7025
        filename_i = dump_fake_images_folder + "/{}_{}.png".format(k, label_i)
        os.makedirs(os.path.dirname(filename_i), exist_ok=True)
        image_i = fake_images[k].astype(np.uint8)
        # image_i = ((image_i*0.5+0.5)*255.0).astype(np.uint8)
        image_i_pil = Image.fromarray(image_i.transpose(1, 2, 0))
        image_i_pil.save(filename_i)