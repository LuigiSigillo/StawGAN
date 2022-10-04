from PIL import Image
import torchvision as tv
import kornia as K
import torch
import numpy as np
import matplotlib.pyplot as plt
grayscale = tv.transforms.Grayscale(num_output_channels=1)
image = Image.open('dataset/train/trainimg/00001.jpg')
imager = Image.open('dataset/train/trainimgr/00001.jpg')
im = torch.tensor(np.asarray(image))
im = im.permute(2,0,1)
lab_image = K.color.rgb_to_lab(im/255.0)
lab_img_np = lab_image.numpy().transpose(1,2,0)
# print(image.size) # Output: (1920, 1280)
# #cropped_image = image.resize((640, 512))
# box = (100, 100, 740, 612)

# cropped_image = image.crop(box)
# print(cropped_image.size) # Output: (1920, 1280)
print(lab_image.shape)
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
rgbr = K.color.lab_to_rgb(lab_image)
img_np_rgb = rgbr.numpy().transpose(1,2,0)

# print(rgbr.shape)
# plt.subplot(255)
# plt.imshow(rgbr, )


import matplotlib.pyplot as plt

count = 1
NSAMPLE=2
fig = plt.figure(figsize=(12,3*NSAMPLE))
for rgb in [lab_img_np, img_np_rgb]:
    ## This section plot the original rgb
    ax = fig.add_subplot(NSAMPLE,4,count)
    ax.imshow(rgb); ax.axis("off")
    ax.set_title("original LAB")
    count += 1
    
    for i, lab in enumerate(["L","A","B"]):
        crgb = np.zeros(rgb.shape)
        crgb[:,:,i] = rgb[:,:,i]
        ax = fig.add_subplot(NSAMPLE,4,count)
        ax.imshow(crgb); ax.axis("off")
        ax.set_title(lab)
        count += 1
    
plt.savefig("aa")




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


