from codecs import ignore_errors
import os
import random
import shutil
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from dataloader import DefaultDataset, DroneVeichleDataset
from utils import getLabel, label2onehot, save_image, save_json
from PIL import Image

from models import LPIPS, InceptionV3
import numpy as np
import glob
import cv2
from scipy import linalg
import subprocess
from ignite.metrics import FID, InceptionScore

device = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_pytorch_fid(args):
    path_real = [args.dataset_path + "/train/trainimg", args.dataset_path + "/train/trainimgr"]
    eval_root = args.eval_dir
    fid_scores = {}
    modals = ('imgr', 'img')
    for p in path_real:
        mod = [m for m in modals if 'train'+m != p.split('/')[-1]]
        print(mod)
        ls = 0
        for src in mod:
            if src=='img':
                eval_path = eval_root +"/val"+ src + "_to_valimgr"
                to = "valimgr" 
            elif src=='imgr':
                eval_path = eval_root +"/val"+ src + "_to_valimg"
                to = "valimg" 
            print("evaluating " + src + " to " + to)
            
            x = str(subprocess.check_output(f'python -m pytorch_fid "{p}" "{eval_path}" --device {device} --batch-size {args.eval_batch_size}',
                shell=True))
            x = x.split(' ')[-1][:-3]
            fid_scores["FID/" + src + " to " + to] = float(x)
            ls += float(x)
        fid_scores["FID/" + to+ "_mean"] = ls / len(mod)
    return fid_scores


def fid_ignite(true, pred):
    fid = FID()
    pred = torch.from_numpy(pred).to(device)
    true = torch.from_numpy(true).to(device)
    if len(pred.shape) != 4:
        pred = pred.unsqueeze(1)
    if pred.size(1) != 3:
        pred = pred.repeat(1, 3, 1, 1)

    if len(true.shape) != 4:
        true = true.unsqueeze(1)
    if true.size(1) != 3:
        true = true.repeat(1, 3, 1, 1)
    print(pred.shape, true.shape)
    fid.update([pred, true])

    valid_fid = fid.compute()
    return valid_fid


def inception_score_ignite(pred):
    metric = InceptionScore()
    pred = torch.from_numpy(pred)
    if len(pred.shape) != 4:
        pred = pred.unsqueeze(1)
    if pred.size(1) != 3:
        pred = pred.repeat(1, 3, 1, 1)
    # true_torch = torch.from_numpy(true).unsqueeze(1).repeat(1, 3, 1, 1)
    metric.update(pred)

    valid_is = metric.compute()
    return valid_is



def calculate_ignite_fid(args):
    path_real = [args.dataset_path + "/train/trainimg", args.dataset_path + "/train/trainimgr"]

    eval_root = args.eval_dir
    fid_scores = {}
    modals = ('imgr', 'img')

    for p in path_real:
        mod = [m for m in modals if 'train'+m != p.split('/')[-1]]
        ls = 0
        for src in mod:
            if src=='img':
                eval_path = eval_root +"/val"+ src + "_to_valimgr" 
                to = "valimgr"
            else:
                eval_path = eval_root +"/val"+ src + "_to_valimg"
                to = "valimg"
 
            print("evaluating " + src + " to " + to)
            print("path predictions:", eval_path, "\n path true:", p)
            pred = jpg_series_reader(eval_path)
            true = jpg_series_reader(p, pred.shape[0]) 
            x = fid_ignite(true, pred)

            fid_scores["FID-ignite/" + src + " to " + to] = float(x)
            ls += float(x)
        fid_scores["FID-ignite/" + to + "_mean"] = ls / len(mod)
    return fid_scores


def calculate_ignite_inception_score(args):
    path_real = [args.dataset_path + "/train/trainimg", args.dataset_path + "/train/trainimgr"]

    eval_root = args.eval_dir
    fid_scores = {}
    modals = ('imgr', 'img')

    for p in path_real:
        mod = [m for m in modals if 'train'+m != p.split('/')[-1]]
        print(mod)
        ls = 0
        for src in mod:
            if src=='img':
                eval_path = eval_root +"/val"+ src + "_to_valimgr" 
                to = "valimgr"
            else:
                eval_path = eval_root +"/val"+ src + "_to_valimg" 
                to = "valimg"
            print("evaluating " + src + " to " + to)
            pred = jpg_series_reader(eval_path)
            x = inception_score_ignite(pred)
            fid_scores["IS-ignite/" + src + " to " +to ] = float(x)
            ls += float(x)
        fid_scores["IS/" + to + "_mean"] = ls / len(mod)
    return fid_scores


'''
Stargan v2 metrics
'''


@torch.no_grad()
def generate_images_fid(netG, args, mode, eval_dataset_imgr, eval_dataset_img, device):
    print('Calculating evaluation metrics...')
    domains = ["valimg","valimgr"]
    domains.sort()
    num_domains = len(domains)
    # calculate_fid_for_all_tasks(args, domains, step=step, mode=mode)
    print('Number of domains: %d' % num_domains)
    loaders = {
        "valimgr_loader": DataLoader(eval_dataset_imgr, batch_size=args.eval_batch_size),
        "valimg_loader": DataLoader(eval_dataset_img, batch_size=args.eval_batch_size),
    }
    mod = {"valimgr": 0, "valimg": 1}

    # loaders = (syneval_dataset, syneval_dataset2,syneval_dataset3)
    # loaders = (DataLoader(syneval_dataset,batch_size=4), DataLoader(syneval_dataset2,batch_size=4),DataLoader(syneval_dataset3,batch_size=4))

    for trg_idx, trg_domain in enumerate(domains):
        src_domains = [x for x in domains if x != trg_domain]
        loader_ref = loaders[trg_domain + "_loader"]
        # path_ref = os.path.join(args.dataset_path + '/val', trg_domain)
        # loader_ref = get_eval_loader(root=path_ref,
        #                                  img_size=args.image_size,
        #                                  batch_size=args.eval_batch_size,
        #                                  imagenet_normalize=False,
        #                                  drop_last=True)
        for src_idx, src_domain in enumerate(src_domains):
            loader_src = loaders[src_domain + "_loader"]
            # path_src = os.path.join(args.dataset_path + '/val', src_domain)
            # loader_src = get_eval_loader(root=path_src,
            #                              img_size=args.image_size,
            #                              batch_size=args.eval_batch_size,
            #                              imagenet_normalize=False)
            task = '%s_to_%s' % (src_domain, trg_domain)
            path_fake = os.path.join(args.eval_dir, task)
            path_fake_gt = os.path.join(path_fake,"ground_truth")
            shutil.rmtree(path_fake, ignore_errors=True)
            os.makedirs(path_fake)
            os.makedirs(path_fake_gt)
            print('Generating images for %s...' % task)
            for i, (x_src) in enumerate(tqdm(loader_src, total=len(loader_src))):
                N = x_src.size(0)
                x_src = x_src.to(device)

                # y_trg = torch.tensor([trg_idx] * N).to(device)
                group_of_images = []
                try:
                    x_ref = next(iter_ref).to(device)
                except:
                    iter_ref = iter(loader_ref)
                    x_ref = next(iter_ref).to(device)

                if x_ref.size(0) > N:
                    x_ref = x_ref[:N]
                
                # idx = random.choice([0,1,2]) #parte da zoomare?
                # x_src_batch = x_src.unsqueeze(0)
                idx = mod[trg_domain]
                c = getLabel(x_src, device, idx, args.c_dim)
                # x_src = x_src[:, :1, :, :]
                x_fake = netG(x_src, None, c, mode='test', wav_type=args.wavelet_type)
                group_of_images.append(x_fake)
                
                # save generated images to calculate FID later
                for k in range(N):
                    filename = os.path.join(path_fake,'%.4i.png' % (i * args.eval_batch_size + (k + 1)))
                    filename_src = os.path.join(path_fake_gt,'%.4i.png' % (i * args.eval_batch_size + (k + 1)))
                    save_image(x_fake[k], ncol=1, filename=filename, args=args)
                    save_image(x_src[k], ncol=1, filename=filename_src, args=args)

                # lpips_value = calculate_lpips_given_images(group_of_images)
                # lpips_values.append(lpips_value)
            # print(lpips_values)
            # calculate LPIPS for each task (e.g. cat2dog, dog2cat)
            # lpips_mean = np.array(lpips_values).mean()
            # lpips_dict['LPIPS_%s/%s' % (mode, task)] = lpips_mean

        # delete dataloaders
        del loader_src
        if mode == 'test':
            del loader_ref
            del iter_ref

    # calculate the average LPIPS for all tasks
    # lpips_mean = 0
    # for _, value in lpips_dict.items():
    #     lpips_mean += value / len(lpips_dict)
    # lpips_dict['LPIPS_%s/mean' % mode] = lpips_mean

    # report LPIPS values
    # filename = os.path.join(args.eval_dir, 'LPIPS_%.5i_%s.json' % (step, mode))
    # save_json(lpips_dict, filename)
    # calculate and report fid values
    




@torch.no_grad()
def calculate_lpips_given_images(group_of_images):
    # group_of_images = [torch.randn(N, C, H, W) for _ in range(10)]
    lpips = LPIPS().eval().to(device)
    lpips_values = []
    num_rand_outputs = len(group_of_images)

    # calculate the average of pairwise distances among all random outputs
    for i in range(num_rand_outputs - 1):
        for j in range(i + 1, num_rand_outputs):
            lpips_values.append(lpips(group_of_images[i].repeat(1, 3, 1, 1), group_of_images[j].repeat(1, 3, 1, 1)))

    lpips_value = torch.mean(torch.stack(lpips_values, dim=0))
    return lpips_value.item()


'''
FID
'''


def calculate_fid_for_all_tasks(args, domains, step, mode):
    print('Calculating FID for all tasks...')
    fid_values = OrderedDict()
    for trg_domain in domains:
        src_domains = [x for x in domains if x != trg_domain]

        for src_domain in src_domains:
            task = '%s_to_%s' % (src_domain, trg_domain)
            path_real = os.path.join(args.dataset_path + "/train", trg_domain.replace("val", "train"))
            path_fake = os.path.join(args.eval_dir, task)
            print('Calculating FID for %s...' % task)
            fid_value = calculate_fid_given_paths(
                paths=[path_real, path_fake],
                img_size=args.img_size,
                batch_size=args.eval_batch_size
            )
            fid_values['FID_%s/%s' % (mode, task)] = fid_value

    # calculate the average FID for all tasks
    fid_mean = 0
    for _, value in fid_values.items():
        fid_mean += value / len(fid_values)
    fid_values['FID_%s/mean' % mode] = fid_mean

    # report FID values
    filename = os.path.join(args.eval_dir, 'FID_%.5i_%s.json' % (step, mode))
    save_json(fid_values, filename)
    return fid_values



def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu - mu2) ** 2) + np.trace(cov + cov2 - 2 * cc)
    return np.real(dist)


@torch.no_grad()
def calculate_fid_given_paths(paths, img_size=256, batch_size=32):
    print('Calculating FID given paths %s and %s...' % (paths[0], paths[1]))
    inception = InceptionV3().eval().to(device)
    loaders = [get_eval_loader(path, img_size, batch_size) for path in paths]
    print(paths)
    mu, cov = [], []
    for i, loader in enumerate(loaders):
        actvs = []
        # print(paths[i])
        for x in tqdm(loader, total=len(loader)):
            try:
                sz = x.size(1)
                if sz == 1:
                    actv = inception(x.repeat(1, 3, 1, 1).to(device))
                elif sz == 3:
                    actv = inception(x.to(device))
                else:
                    raise Exception("check FID dim")
            except:
                sz = x[0].size(1)
                if sz == 1:
                    actv = inception(x[0].repeat(1, 3, 1, 1).to(device))
                elif sz == 3:
                    actv = inception(x[0].to(device))
                else:
                    raise Exception("check FID dim")

            actvs.append(actv)
        actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
        mu.append(np.mean(actvs, axis=0))
        cov.append(np.cov(actvs, rowvar=False))
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    return fid_value


def get_eval_loader(path, image_size, batch_size):
    return DataLoader(DefaultDataset(path, img_size=image_size), batch_size=batch_size)


def DICE(Vref, Vseg):
    dice = 2 * (Vref & Vseg).sum() / (Vref.sum() + Vseg.sum()) * 100
    return dice




def png_series_reader(dir):
    V = []
    png_file_list = glob.glob(dir + '/*.png')
    png_file_list.sort()
    for filename in png_file_list:
        image = cv2.imread(filename, 0)
        V.append(image)
    V = np.array(V, order='A')
    #V = V.astype(bool)
    return V

def jpg_series_reader(dir, mlen=None):
    V = []
    jpg_file_list = glob.glob(dir + '/*.jpg')
    png_file_list = glob.glob(dir + '/*.png')
    tot_list = (png_file_list+jpg_file_list)
    random.shuffle(tot_list)
    box = (100, 100, 740, 612)
    if mlen!=None:
        tot_list = tot_list[:mlen]
    for filename in tqdm(tot_list):
        img = Image.open(filename).convert("RGB")
        img = img.crop(box)
        img = img.resize((256, 256))
        img = np.asarray(img)
        V.append(img.transpose(2, 0, 1))
    V = np.array(V, order='A')
    return V



def create_images_for_dice_or_s_score(args, netG, idx_eval, syneval_loader, dice_=False, calculate_mae=False):
    subp ="dice" if dice_ else "s_score"
    shutil.rmtree(args.eval_dir+"/Segmentation/"+subp, ignore_errors=True)
    shutil.rmtree(args.eval_dir+"/Ground/"+subp, ignore_errors=True)
    os.makedirs(args.eval_dir+"/Segmentation/"+subp)
    os.makedirs(args.eval_dir+"/Ground/"+subp)
    output_mae, plotted = 0, 0 
    mae = nn.L1Loss()
    with torch.no_grad():
        for epoch, (x_real, t_img, shape_mask, mask, label_org) in tqdm(enumerate(syneval_loader), total=len(syneval_loader)):

            # label_trg = label_org[rand_idx]
            c_org = label2onehot(label_org, args.c_dim)
            # c_trg = label2onehot(label_trg, args.c_dim)
            x_real = x_real.to(device)  # Input images.
            c_org = c_org.to(device)  # Original domain labels.
            # c_trg = c_trg.to(device)
            t_img = t_img.to(device)
            # translate only in one domain
            # if dice_:
            # s = c_trg.size(0)
            # c_trg = c_trg[:s]
            c_trg = getLabel(x_real, device, idx_eval, args.c_dim)
            # Original-to-target domain.

            if not dice_:
                c_t, x_r, t_i, c_o = [], [], [], []
                for i, x in enumerate(c_trg):
                    if not torch.all(x.eq(c_org[i])):
                        c_t.append(x)
                        x_r.append(x_real[i])
                        t_i.append(t_img[i])
                        c_o.append(c_org[i])

                    # print(x,c_org[i])
                if len(c_t) == 0:
                    continue
                c_trg = torch.stack(c_t, dim=0).to(device)
                x_real = torch.stack(x_r, dim=0).to(device)
                t_img = torch.stack(t_i, dim=0).to(device)
                c_org = torch.stack(c_o, dim=0).to(device)

            # good for dice
            x_fake, t_fake = netG(x_real, t_img,
                                        c_trg, wav_type=args.wavelet_type)  # G(image,target_image,target_modality) --> (out_image,output_target_area_image)

            if not dice_:
                try:
                    _, t_reconst = netG(x_fake, t_fake, c_org,wav_type=args.wavelet_type)
                except:
                    d = args.device
                    args.device = 'cpu'
                    _, t_reconst = netG.cpu()(x_fake.cpu(), t_fake.cpu(), c_org.cpu())
                    args.device = d
                t_fake = t_reconst.to(device)
            # Target-to-original domain.
            # fig = plt.figure(dpi=120)
            # with torch.no_grad():
            #     if plotted == 0:
            #         plt.subplot(241)
            #         plt.imshow(denorm(x_real[0]).squeeze().cpu().numpy(), cmap='gray')
            #         plt.title("original image")
            #         plt.subplot(242)
            #         plt.imshow(denorm(x_fake[0]).squeeze().cpu().numpy(), cmap='gray')
            #         plt.title("fake image")
            #         # plt.subplot(253)
            #         # plt.imshow(denorm(x_reconst[0]).squeeze().cpu().numpy(), cmap='gray')
            #         # plt.title("x reconstruct image")
            #         plt.subplot(243)
            #         plt.imshow(denorm(t_img[0]).squeeze().cpu().numpy(), cmap='gray')
            #         plt.title("original target")
            #         plt.subplot(244)
            #         plt.imshow(denorm(t_fake[0]).squeeze().cpu().numpy(), cmap='gray')
            #         plt.title("fake target")
            #         plt.show()
            #         plotted = 1
            #         plt.close(fig)

            if calculate_mae:
                output_mae += mae(t_fake, t_img)
            for k in range(c_trg.size(0)):
                filename = os.path.join(args.eval_dir,"Segmentation",subp,
                                        '%.4i.png' % (epoch * args.eval_batch_size + (k + 1)))
                save_image(t_fake[k], ncol=1, filename=filename, args=args)
                filename = os.path.join(args.eval_dir,"Ground",subp,
                                        '%.4i.png' % (epoch * args.eval_batch_size + (k + 1)))
                # if t_img.size(1) == 5:
                #     t_img = t_img[:, :1]

                save_image(t_img[k], ncol=1, filename=filename, args=args)
    return output_mae/len(syneval_loader)




def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu - mu2) ** 2) + np.trace(cov + cov2 - 2 * cc)
    return np.real(dist)


def _thresh(img):
    img[img > 0.5] = 1
    img[img <= 0.5] = 0
    return img


def IoU(y_pred, y_true):
    y_pred = _thresh(y_pred)
    y_true = _thresh(y_true)

    intersection = np.logical_and(y_pred, y_true)
    union = np.logical_or(y_pred, y_true)
    if not np.any(union):
        return 0 if np.any(y_pred) else 1
    iou = intersection.sum() / float(union.sum())
    return iou, np.mean(iou)


def compute_miou(validation_pred, validation_true):
    # Compute mIoU         
    validation_pred_np = np.asarray(validation_pred)
    validation_true_np = np.asarray(validation_true)
    # validation_pred_torch = torch.from_numpy(validation_pred_np)
    # validation_true_torch = torch.from_numpy(validation_true_np)
    # print("Val pred", validation_pred_torch.shape)
    # print("Val true", validation_true_torch.shape)
    iou, miou = IoU(validation_pred_np, validation_true_np)

    return iou

def calculate_metrics_segmentation(args, net_G):
    mod = ['imgr', 'img']
    dice_dict, s_score_dict, iou_dict, mae_dict = {}, {}, {}, {}
    tot_rep = 2 if args.preloaded_data else 1

    for idx in tqdm(range(tot_rep)):
        if args.preloaded_data:
            syneval_dataset_tot = DroneVeichleDataset(path=args.dataset_path, split='val', colored_data=args.color_images)
            syneval_dataset_tot.load_dataset(path=args.dataset_path+"/tensors",split="val", idx=str(idx), img_size=args.img_size, colored_data=args.color_images)
        else:
            syneval_dataset_tot = DroneVeichleDataset(path=args.dataset_path, split='val', colored_data=args.color_images)
        syneval_loader = DataLoader(syneval_dataset_tot, shuffle=True, batch_size=args.eval_batch_size)    

        for i in tqdm(range(len(mod))):  # 2 domains
            # ======= Directories =======
            ground_dir = os.path.normpath(args.eval_dir+'/Ground/dice')
            seg_dir = os.path.normpath(args.eval_dir+'/Segmentation/dice')
            mae_dict["mae/" + mod[i]+str(idx)] = create_images_for_dice_or_s_score(args, net_G, i, syneval_loader, dice_=True, calculate_mae=True)
            # ======= Volume Reading =======
            Vref = png_series_reader(ground_dir)
            Vseg = png_series_reader(seg_dir)
            print('Volumes imported.')
            # ======= Evaluation =======
            print('Calculating for  modality ...', mod[i]+str(idx))
            dice = DICE(Vref, Vseg)
            dice_dict["DICE/" + mod[i]+str(idx)] = dice

            iou = compute_miou(Vref, Vseg)
            iou_dict["IoU/" + mod[i]+str(idx)] = iou

            # calculate s score
            create_images_for_dice_or_s_score(args, net_G, i, syneval_loader, dice_=False)
            # ======= Volume Reading =======
            ground_dir = os.path.normpath(args.eval_dir+'/Ground/s_score')
            seg_dir = os.path.normpath(args.eval_dir+'/Segmentation/s_score')
            Vref = png_series_reader(ground_dir)
            Vseg = png_series_reader(seg_dir)
            s_score = DICE(Vref, Vseg)
            s_score_dict["S-SCORE/" + mod[i]+str(idx)] = s_score
    
    dice_d, s_score_d, iou_d, mae_d =  {}, {}, {}, {}
    for i in range(len(mod)):
        dice_d["DICE/" + mod[i]] = sum([dice_dict["DICE/" + mod[i]+str(idx)] for idx in range(tot_rep)])/tot_rep
        s_score_d["S-SCORE/" + mod[i]] = sum([s_score_dict["S-SCORE/"+ mod[i]+str(idx)] for idx in range(tot_rep)])/tot_rep
        iou_d["IoU/" + mod[i]] = sum([iou_dict["IoU/" + mod[i]+str(idx)] for idx in range(tot_rep)])/tot_rep
        mae_d["mae/" + mod[i]] = sum([mae_dict["mae/" + mod[i]+str(idx)] for idx in range(tot_rep)])/tot_rep

    return dice_d, s_score_d, iou_d, mae_d 

#TODO
def my_metrics():
    fid_scores={}
    #calculate FID entire translated image
    p = "dataset/train/trainimgr"
    eval_path = "results/color_pretrained_256/valimg_to_valimgr"
    src = "img"
    to = "imgr"
    x = str(subprocess.check_output(f'python -m pytorch_fid "{p}" "{eval_path}" --device {device} --batch-size 50',
                shell=True))
    x = x.split(' ')[-1][:-3]
    fid_scores["FID/" + src + " to " + to] = float(x)
    
    # #calculate FID translated image without targets
    # print()
    #calculate FID translated targets
    p = "results/color_pretrained_256/Ground/dice"
    eval_path = "results/color_pretrained_256/Segmentation/dice"
    src = "img"
    to = "imgr"
    x = str(subprocess.check_output(f'python -m pytorch_fid "{p}" "{eval_path}" --device {device} --batch-size 50',
                shell=True))
    x = x.split(' ')[-1][:-3]
    fid_scores["FID_target/" + src + " to " + to] = float(x)
    #see if FID(trans_targ) > FID(trans_without_t)
    print(fid_scores)


def calculae_metrics_translation(args, net_G):
    eval_dataset_imgr, eval_dataset_img = DefaultDataset(args.dataset_path+"/val/valimgr"), \
                                         DefaultDataset(args.dataset_path+"/val/valimg")
    generate_images_fid(net_G, args, 'test',
                                      eval_dataset_imgr,
                                      eval_dataset_img,
                                      device
                                      )
    #fid_stargan = calculate_fid_for_all_tasks(args, domains = ["valimg","valimgr"], step=args.epoch*10, mode="stargan")
    fid_dict = calculate_pytorch_fid(args)
    fid_ignite_dict = calculate_ignite_fid(args)
    IS_ignite_dict = calculate_ignite_inception_score(args)
    
    return {}, fid_dict, IS_ignite_dict, fid_ignite_dict 


def calculate_all_metrics(args, net_G, device="cuda" if torch.cuda.is_available() else "cpu"):
    args.eval_dir=os.path.join(args.eval_dir, args.experiment_name)
    os.makedirs(args.eval_dir, exist_ok=True)

    fid_stargan, fid_dict, IS_ignite_dict, fid_ignite_dict = calculae_metrics_translation(args, net_G)
    dice_dict, s_score_dict, iou_dict, mae_dict = calculate_metrics_segmentation(args, net_G)
    return fid_stargan, fid_dict, dice_dict, s_score_dict, iou_dict, IS_ignite_dict, fid_ignite_dict, mae_dict



def evaluation(args):
    ii = args.sepoch * 650
    in_c = 1 if not args.color_images else 3
    in_c_gen = in_c+4 if args.wavelet_type != None else in_c
    if not args.real:
        from models_quat import Generator
        while (in_c_gen + args.c_dim) % 4 != 0: #3+2
            in_c_gen+=1
        net_G = Generator(in_c=in_c_gen + args.c_dim, mid_c=args.G_conv, layers=2, s_layers=3, affine=True, last_ac=True,
                     colored_input=args.color_images, wav=args.wavelet_type,real=args.real, qsn=args.qsn, phm=args.phm).to(device)
    else:
        from models import Generator
        net_G = Generator(in_c=in_c_gen + args.c_dim, mid_c=args.G_conv, layers=2, s_layers=3, affine=True, last_ac=True,
                            colored_input=args.color_images, wav=args.wavelet_type).to(device)
    
    ep = str(args.epoch) if not args.preloaded_data else str(args.epoch*10)
    net_G.load_state_dict(torch.load(args.save_path+"/"+args.experiment_name+"/netG_use_"+ep+"_"+args.experiment_name+".pkl", map_location=device))
    with wandb.init(config=args, project="targan_drone") as run:
        wandb.run.name = args.experiment_name
        fid_stargan, fid_dict, dice_dict, s_score_dict, iou_dict, IS_ignite_dict, fid_ignite_dict, mae_dict = calculate_all_metrics(args, net_G)


        wandb.log(dict(fid_stargan), step=ii + 1, commit=False)
        wandb.log(dict(fid_dict), step=ii + 1, commit=False)
        wandb.log(dict(dice_dict), step=ii + 1, commit=False)
        wandb.log(dict(s_score_dict), step=ii + 1, commit=False)
        wandb.log(dict(IS_ignite_dict), step=ii + 1, commit=False)
        wandb.log(dict(fid_ignite_dict), step=ii + 1, commit=False)
        wandb.log(dict(mae_dict), step=ii + 1, commit=False)
        wandb.log(iou_dict, commit=True)


