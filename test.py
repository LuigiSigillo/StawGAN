from PIL import Image
import torchvision as tv
import kornia as K
import torch
import numpy as np
import sys
import cv2

from models_quat import Generator, DiscriminatorStyle, StyleEncoder

# from dataloader import label_preprocess
# np.set_printoptions(threshold=sys.maxsize)

# import matplotlib.pyplot as plt
# grayscale = tv.transforms.Grayscale(num_output_channels=1)
# image = Image.open('dataset/train/trainimg/00001.jpg')
# imager = Image.open('dataset/train/trainimgr/00001.jpg')


# color_palette = [(255,0,0), (128,0,128), (0,192,0), (0,100,0), (200,200,0)]
# dict_palette = {"car":color_palette[0], 
#                 "feright_car": color_palette[1],
#                 "truck": color_palette[2],
#                 "bus" :color_palette[3],
#                 "van":color_palette[4]}

# threesolds = [(240,255), (113,145), (177,207), (85,115), (185,215)]
# split='val'
# i = 'dataset/val/valimg/01469.jpg'
# raw_classes, classes_temp=[],[]
# box = (100, 100, 740, 612)
# # image_maskr = Image.open('dataset/val/valmaskscolr/00021.jpg').convert("RGB")
# for idx in range(6):
#     a =  i.replace(split+"img",split+"maskscol")
#     try:
#         img_segm = Image.open(a.replace('.jpg', '_'+str(idx+1)+'.jpg')).convert('L')
#     except:
#         continue
#     img_segm = img_segm.crop(box)

#     # convert image to numpy array
#     img_segm = np.asarray(img_segm)
#     img_segm = cv2.resize(img_segm, (256, 256), interpolation=cv2.INTER_LINEAR)

#     test = label_preprocess(img_segm)
#     print(idx+1)
#     c = np.array(idx+1) #class
#     classes_temp.append((test,c))

# raw_classes.append(classes_temp)
    # plt.imshow(test,cmap='gray')
    # plt.show()


# for i,(min_t,max_t) in enumerate(threesolds):
#     new_a = np.zeros(np_img.shape)
#     print(np.count_nonzero( (min_t<np_img) & (np_img<max_t) ) )
#     new_a[(min_t<np_img) & (np_img<max_t)] = color_palette[i][0] if color_palette[i][0]!= 0 else color_palette[i][1]
#     # print(new_a)

#     test = Image.fromarray(new_a.astype("uint8"))
#     test.save('rr'+list(dict_palette.keys())[i]+".jpg")


# blackPxNum = np.count_nonzero([np.asarray(img)<=76]) #number of black pixels
# whitePxNum = np.asarray(img).size - blackPxNum 
# print(whitePxNum)
# print(blackPxNum)
# if whitePxNum<100:
#     """segment is black"""
#     print("blac")
# im = torch.tensor(np.asarray(image))
# im = im.permute(2,0,1)
# lab_image = K.color.rgb_to_lab(im/255.0)
# lab_img_np = lab_image.numpy().transpose(1,2,0)
# print(image.size) # Output: (1920, 1280)
# #cropped_image = image.resize((640, 512))
# box = (100, 100, 740, 612)

# cropped_image = image.crop(box)
# print(cropped_image.size) # Output: (1920, 1280)
# print(lab_image.shape)
# grayscale(image).save('grayscale.jpg')
# imager.save('rr.jpg')
# img_np = np.asarray(image)

# print(img_np.shape)

# plt.subplot(251)
# plt.imshow(img_np[:,:,0]/255.0 )
# plt.subplot(252)
# plt.imshow(img_np[:,:,1]/255.0, )
# plt.subplot(253)
# plt.imshow(img_np[:,:,2]/255.0, )
# plt.subplot(254)
# plt.imshow(img_np, )
# rgbr = K.color.lab_to_rgb(lab_image)
# img_np_rgb = rgbr.numpy().transpose(1,2,0)

# print(rgbr.shape)
# plt.subplot(255)
# plt.imshow(rgbr, )


# import matplotlib.pyplot as plt

# count = 1
# NSAMPLE=2
# fig = plt.figure(figsize=(12,3*NSAMPLE))
# for rgb in [lab_img_np, img_np_rgb]:
#     ## This section plot the original rgb
#     ax = fig.add_subplot(NSAMPLE,4,count)
#     ax.imshow(rgb); ax.axis("off")
#     ax.set_title("original LAB")
#     count += 1
    
#     for i, lab in enumerate(["L","A","B"]):
#         crgb = np.zeros(rgb.shape)
#         crgb[:,:,i] = rgb[:,:,i]
#         ax = fig.add_subplot(NSAMPLE,4,count)
#         ax.imshow(crgb); ax.axis("off")
#         ax.set_title(lab)
#         count += 1
    
# plt.savefig("aa")




# import os
# import numpy as np
# root = "dataset/kaist-cvpr15/images"
# # setss = os.listdir(root)
# # for sett in setss:
# #     vvs = os.listdir(os.path.join(root,sett))
# #     for vv in vvs:
# with open("dataset/kaist-cvpr15/imageSets/train-all-04.txt") as f:
#     llp = [line.strip() for line in f.readlines()]
# print(len(llp))
# path1,path2, raw_path = [],[],[]
# for sett in llp:
#     splitted = sett.split("/")
#     splitted.insert(-1,"lwir")
#     final = "/".join(splitted) +".jpg"
#     path1.append(final)
#     path2.append(final.replace("lwir","visible"))
# list_path = sorted(path1+path2)

# print(len(list_path))
# for x in list_path:
#     if "lwir" in x:
#         c = np.array(0) #infrared
#         # tmp =  x.replace(split+"imgr",split+"masksr")
#     elif "visible" in x:
#         c = np.array(1)
#         # tmp =  x.replace(split+"img",split+"masks")
#     else:
#         raise Exception('wrong path probably')
#     raw_path.append([x,c])

# print(raw_path[2])        


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    for i in range(batch_size):
        out[i, labels[i].long()] = 1
    return out
import random
def compute_g_loss(nets, args, x_real, t_img, c_trg, y_trg, x_segm=None):
    
    # estrapolo lo style code dalla immagine segmentata/quella intera?
    # gli passo anche la lebel target (quindi se é car car?)
    netSE = StyleEncoder()
    netG = Generator(in_c=3 + 2, mid_c=64, layers=2, s_layers=3, affine=True, last_ac=True,
                     colored_input=True, wav=None)
    netD_style = DiscriminatorStyle()
    yy_trg = torch.tensor([1., 0., 0., 0., 0.])
    while torch.equal(yy_trg,torch.tensor([1., 0., 0., 0., 0.])):
        idx = random.randint(0,5)
        yy_trg = y_trg[idx]
    
    s_trg = netSE(x_segm[:,idx], yy_trg.unsqueeze(0).long())
    
    #genero l immagine fake passandogli la segmentate/reale di un truck con lo stylecode della car
    x_fake, t_fake = netG(x_real, t_img, c_trg, style = s_trg, wav_type=None)
    
    
    #calcolo la loss passando al dsicriminatore la label target car e la fake
    out = netD_style(x_fake, yy_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    #estgrapolo stile da fake img segm passando sempre stesso target car??
    s_pred = netSE(x_fake, yy_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    # s_trg2 = nets.style_encoder(x_ref2, y_trg)
    # x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    # x_fake2 = x_fake2.detach()
    # loss_ds = torch.mean(torch.abs(x_fake - x_fake2))


    #USARE QUELLA CHE GIA CI STA
    # cycle-consistency loss
    # s_org = netSE(x_real, y_org)
    # x_rec = netG(x_fake, style=s_org, )
    # loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty #+ args.lambda_cyc * loss_cyc
    return loss, [ loss_adv.item(),
                    loss_sty.item(),]
                    #loss_cyc.item()]

import torch.nn.functional as F
def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss
#torch.Size([3, 256, 256]) torch.Size([3, 256, 256]) torch.Size([3, 256, 256]) torch.Size([1, 256, 256]) torch.Size([]) torch.Size([6, 3, 256, 256]) torch.Size([6])
# 1. Preprocess input data
# Generate target domain labels randomly.
label_org = torch.tensor([0])
classes_org = torch.tensor([1,3,2,0,0,0])
rand_idx = torch.randperm(label_org.size(0))
rand_idx_classes = torch.randperm(classes_org.size(0))

label_trg = label_org[rand_idx]
classes_trg = classes_org[rand_idx_classes]

c_org = label2onehot(label_org, 2)
c_trg = label2onehot(label_trg, 2)
d_false_org = label2onehot(label_org + 2, 2 * 2)
d_org = label2onehot(label_org,2* 2)
g_trg = label2onehot(label_trg, 2 * 2)


c_classes_org = label2onehot(classes_org, 5)
c_classes_trg = label2onehot(classes_trg, 5)
d_false_classes_org = label2onehot(classes_org + 2, 5 *2)
d_classes_org = label2onehot(classes_org,5* 2)
g_classes_trg = label2onehot(classes_trg, 5 * 2)

# # train the generator
g_loss, g_losses_latent = compute_g_loss(None, None, 
                                    x_real=torch.randn(1,3,256,256), t_img=torch.randn(1,3,256,256), y_trg=c_classes_org, c_trg=c_trg, x_segm=torch.randn(1,6,3,256,256) )
# self._reset_grad()
# g_loss.backward()
# optims.generator.step()
# optims.style_encoder.step()


