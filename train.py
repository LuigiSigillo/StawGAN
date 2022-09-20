from models import Generator,Discriminator,ShapeUNet
from dataloader import *
from torch.utils.data import DataLoader
from utils import *
import argparse
import time
import matplotlib.pyplot as plt
import datetime
import wandb
import copy
import torch.nn.functional as F
from tqdm import tqdm


def train(args):
    glr = args.lr
    dlr = args.ttur
    print(glr, dlr)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    # set_seed(args.random_seed)
    if not args.preloaded_data:
        syn_dataset = DroneVeichleDataset(split='train', colored_data=args.color_images)
        syn_loader = DataLoader(syn_dataset, batch_size=args.batch_size, shuffle=True)
        syneval_dataset = DroneVeichleDataset(split='val', colored_data=args.color_images)
    else:
        idx = 0
        syn_dataset = DroneVeichleDataset(to_be_loaded=True)
        syn_dataset.load_dataset("dataset/tensors",split="train", idx=str(idx), img_size=args.img_size, colored_data=args.color_images)
        syn_loader = DataLoader(syn_dataset, batch_size=args.batch_size, shuffle=True)

        syneval_dataset = DroneVeichleDataset(to_be_loaded=True)
        syneval_dataset.load_dataset("dataset/tensors",split="val", idx=str(idx), img_size=args.img_size, colored_data=args.color_images)

    in_c = 1 if not args.color_images else 3
    netG = Generator(in_c= in_c + args.c_dim, mid_c=args.G_conv, layers =2, s_layers=3, affine=True, last_ac=True,
                        colored_input=args.color_images)
    netH = ShapeUNet(img_ch=in_c, output_ch=1, mid=args.h_conv)
    netD_i = Discriminator(c_dim=args.c_dim * 2, image_size=args.img_size, colored_input=args.color_images)
    netD_t = Discriminator(c_dim=args.c_dim * 2, image_size=args.img_size, colored_input=args.color_images)


    g_optimizier = torch.optim.Adam(netG.parameters(), lr=glr, betas=(args.betas[0], args.betas[1]))
    di_optimizier = torch.optim.Adam(netD_i.parameters(), lr=dlr, betas=(args.betas[0], args.betas[1]))
    dt_optimizier = torch.optim.Adam(netD_t.parameters(), lr=dlr, betas=(args.betas[0], args.betas[1]))
    h_optimizier = torch.optim.Adam(netH.parameters(), lr=glr, betas=(args.betas[0], args.betas[1]))

    netG_use = copy.deepcopy(netG)
    netG.to(device)
    netD_i.to(device)
    netD_t.to(device)
    netH.to(device)
    netG_use.to(device)

    start_time = time.time()
    print('start training...')

    ii = 0
    # logdir = "log/" + args.save_path
    # log_writer = LogWriter(logdir)
    with wandb.init(config=args, project="targan_drone") as run:
        wandb.run.name = args.experiment_name
        for i in tqdm(range(args.sepoch,args.epoch)):
            for epoch, (x_real, t_img, shape_mask, mask, label_org) in tqdm(enumerate(syn_loader), total=len(syn_loader)):
                # 1. Preprocess input data
                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]
                c_org = label2onehot(label_org, args.c_dim)
                c_trg = label2onehot(label_trg, args.c_dim)
                d_false_org = label2onehot(label_org + args.c_dim, args.c_dim * 2)
                d_org = label2onehot(label_org, args.c_dim * 2)
                g_trg = label2onehot(label_trg, args.c_dim * 2)
                # plt.subplot(231)
                # plt.imshow(  x_real[2].cpu().detach().permute(1, 2, 0).numpy()  )
                x_real = x_real.to(device)  # Input images.

                c_org = c_org.to(device)  # Original domain labels.
                c_trg = c_trg.to(device)  # Target domain labels.
                d_org = d_org.to(device)  # Labels for computing classification loss.
                g_trg = g_trg.to(device)  # Labels for computing classification loss.
                d_false_org = d_false_org.to(device) # Labels for computing classification loss.
                mask = mask.to(device)
                # shape_mask = shape_mask.to(device)

                t_img = t_img.to(device)
                # plt.subplot(232)
                # plt.imshow(t_img[2].cpu().detach().permute(1, 2, 0).numpy())

                index = loss_filter(mask)
                # 2. Train the discriminator
                # Compute loss with real whole images.
                out_src, out_cls = netD_i(x_real)
                # plt.subplot(233)
                # plt.imshow(  x_real[2].cpu().detach().permute(1, 2, 0).numpy(), cmap='gray')
                # plt.savefig('x_real greyscaled')
                # raise Exception
                d_loss_real = -torch.mean(out_src)
                d_loss_cls = F.binary_cross_entropy_with_logits(out_cls, d_org, reduction='sum') / out_cls.size(0)

                # Compute loss with fake whole images.
                with torch.no_grad():
                    x_fake, t_fake = netG(x_real, t_img, c_trg)
                # plt.imshow(  x_fake[2].cpu().detach().permute(1, 2, 0).numpy(), cmap='gray')
                # plt.savefig('x-fake greyscaled')
                # raise Exception
                out_src, out_f_cls = netD_i(x_fake.detach())
                d_loss_fake = torch.mean(out_src)
                d_loss_f_cls = F.binary_cross_entropy_with_logits(out_f_cls, d_false_org, reduction='sum') / out_f_cls.size(0)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1, 1).to(device)
                x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
                out_src, _ = netD_i(x_hat)
                d_loss_gp = gradient_penalty(out_src, x_hat, device)

                # compute loss with target images
                if index.shape[0] != 0:
                    out_src, out_cls = netD_t(torch.index_select(t_img, dim=0, index=index))
                    d_org = torch.index_select(d_org, dim=0, index=index)
                    d_loss_real_t = -torch.mean(out_src)
                    d_loss_cls_t = F.binary_cross_entropy_with_logits(out_cls, d_org, reduction='sum') / out_cls.size(0)

                    out_src, out_f_cls = netD_t(torch.index_select(t_fake.detach(), dim=0, index=index))
                    d_false_org = torch.index_select(d_false_org, dim=0, index=index)
                    d_loss_fake_t = torch.mean(out_src)
                    d_loss_f_cls_t = F.binary_cross_entropy_with_logits(out_f_cls, d_false_org,
                                                                        reduction='sum') / out_f_cls.size(0)

                    x_hat = (alpha * t_img.data + (1 - alpha) * t_fake.data).requires_grad_(True)
                    x_hat = torch.index_select(x_hat, dim=0, index=index)
                    out_src, _ = netD_t(x_hat)
                    d_loss_gp_t = gradient_penalty(out_src, x_hat, device)

                    dt_loss = d_loss_real_t + d_loss_fake_t + d_loss_cls_t + d_loss_gp_t * 10 + d_loss_f_cls_t * args.w_d_false_t_c
                    w_dt = (-d_loss_real_t - d_loss_fake_t).item()
                else:
                    dt_loss = torch.FloatTensor([0]).to(device)
                    w_dt = 0
                    d_loss_f_cls_t = torch.FloatTensor([0]).to(device)
                # Backward and optimize.
                di_loss = d_loss_real + d_loss_fake + d_loss_cls + d_loss_gp * 10 + d_loss_f_cls * args.w_d_false_c
                d_loss = di_loss + dt_loss
                w_di = (-d_loss_real - d_loss_fake).item()

                g_optimizier.zero_grad()
                di_optimizier.zero_grad()
                dt_optimizier.zero_grad()
                d_loss.backward()
                di_optimizier.step()
                dt_optimizier.step()

                #  3. Train the generator
                # Original-to-target domain.
                x_fake, t_fake = netG(x_real, t_img, c_trg)
                out_src, out_cls = netD_i(x_fake)
                g_loss_fake = -torch.mean(out_src)
                g_loss_cls = F.binary_cross_entropy_with_logits(out_cls, g_trg, reduction='sum') / out_cls.size(0)
                # mask = mask.repeat(1, 3, 1, 1)
                # shape_mask = shape_mask.repeat(1, 3, 1, 1)
                # print(shape_mask.shape,netH(x_fake).shape )
                # shape_loss = F.mse_loss(netH(x_fake), shape_mask.float())
                # Target-to-original domain.
                x_reconst, t_reconst = netG(x_fake, t_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                if index.shape[0] != 0:
                    out_src, out_cls = netD_t(torch.index_select(t_fake, dim=0, index=index))
                    g_trg = torch.index_select(g_trg, dim=0, index=index)
                    g_loss_fake_t = -torch.mean(out_src)
                    g_loss_cls_t = F.binary_cross_entropy_with_logits(out_cls, g_trg, reduction='sum') / out_cls.size(0)
                    gt_loss = g_loss_fake_t + g_loss_cls_t * args.w_g_t_c
                else:
                    gt_loss = torch.FloatTensor([0]).to(device)
                    g_loss_cls_t = torch.FloatTensor([0]).to(device)

                # print(netH(t_fake).shape, mask.shape)

                shape_loss_t = F.mse_loss(netH(t_fake), mask.float()) #mask.repeat(1, 3, 1, 1).float()
                g_loss_rec_t = torch.mean(torch.abs(t_img - t_reconst))
                cross_loss = torch.mean(torch.abs(denorm(x_fake) * mask - denorm(t_fake)))
                # Backward and optimize.
                gi_loss = g_loss_fake + args.w_cycle * g_loss_rec + g_loss_cls * args.w_g_c #+ shape_loss* args.w_shape
                gt_loss = gt_loss + args.w_cycle * g_loss_rec_t + shape_loss_t* args.w_shape + cross_loss * args.w_g_cross
                g_loss = gi_loss + gt_loss

                g_optimizier.zero_grad()
                di_optimizier.zero_grad()
                dt_optimizier.zero_grad()
                h_optimizier.zero_grad()
                g_loss.backward()
                g_optimizier.step()
                h_optimizier.step()

                moving_average(netG, netG_use, beta=0.999)

                if (epoch + 0) % 10 == 0:
                    all_losses = dict()

                    all_losses["train/D/w_di"] = w_di
                    all_losses["train/D/w_dt"] = w_dt
                    all_losses["train/D/loss_f_cls"] = d_loss_f_cls.item()
                    all_losses["train/D/loss_f_cls_t"] = d_loss_f_cls_t.item()
                    all_losses["train/G/loss_cls"] = g_loss_cls.item()
                    all_losses["train/G/loss_cls_t"] = g_loss_cls_t.item()
                    # all_losses["train/G/loss_shape"] = shape_loss.item()
                    all_losses["train/G/loss_shape_t"] = shape_loss_t.item()
                    all_losses["train/G/loss_cross"] = cross_loss.item()
                    wandb.log(all_losses, step=ii, commit=True)

                ii = ii + 1
                ###################################

            if (i + 1) % 1 == 0 and (i + 1) > 0:
                # show syn images after every epoch
                x_real,x_infrared,x_rgb, trg_orig, trg_infra_fake, trg_rgb_fake = plot_images(netG_use, syneval_dataset, device, args.c_dim)
                # print(x.shape, y.shape, z.shape)
                # plt.subplot(231)
                # plt.imshow(  x.cpu().detach().permute(1, 2, 0).numpy()  )
                # plt.subplot(232)
                # plt.imshow(  y.cpu().detach().numpy()  )
                # plt.subplot(233)
                # plt.imshow(  z.cpu().detach().numpy()  )
                # plt.savefig('final')
                wandb.log({"orig": wandb.Image(x_real, caption="orig_" + str(i))}, commit=False)
                wandb.log({"ir": wandb.Image(x_infrared, caption="ir_" + str(i))}, commit=False)
                wandb.log({"img": wandb.Image(x_rgb, caption="img_" + str(i))}, commit=False)
                wandb.log({"orig_trg": wandb.Image(trg_orig, caption="orig_trg_" + str(i))}, commit=False)
                wandb.log({"ir_trg": wandb.Image(trg_infra_fake, caption="ir_trg_" + str(i))}, commit=False)
                wandb.log({"img_trg": wandb.Image(trg_rgb_fake, caption="img_trg_" + str(i))}, commit=False)

                # raise Exception


            if (i + 1) == args.epoch:
                args.net_name = 'netG'
                save_state_net(netG, args, i + 1, g_optimizier, args.experiment_name)
                args.net_name = 'netG_use'
                save_state_net(netG_use, args, i + 1, None, args.experiment_name)
                args.net_name = 'netDi'
                save_state_net(netD_i, args, i + 1, di_optimizier, args.experiment_name)
                args.net_name = 'netDt'
                save_state_net(netD_t, args, i + 1, dt_optimizier, args.experiment_name)
                args.net_name = 'netH'
                save_state_net(netH, args, i + 1, h_optimizier, args.experiment_name)
                # fidstar, fid, dice, ravd, s_score, fid_giov, iou_dict, IS_ignite_dict, fid_ignite_dict, mae_dict = calculate_all_metrics(nets,
                #                                                                     syneval_dataset,
                #                                                                     syneval_dataset2,
                #                                                                     syneval_dataset3,
                #                                                                     syneval_loader)
                # wandb.log(dict(fidstar), step=ii + 1, commit=False)
                # wandb.log(dict(fid), step=ii + 1, commit=False)
                # wandb.log(dict(fid_giov), step=ii + 1, commit=False)
                # wandb.log(dict(IS_ignite_dict), step=ii + 1, commit=False)
                # wandb.log(dict(fid_ignite_dict), step=ii + 1, commit=False)

                # wandb.log(dict(dice), step=ii + 1, commit=False)
                # wandb.log(dict(ravd), step=ii + 1, commit=False)
                # wandb.log(dict(iou_dict), step=ii + 1, commit=False)
                # wandb.log(dict(mae_dict), step=ii + 1, commit=False)
                # wandb.log(dict(s_score), step=ii + 1, commit=False)
                # formatt = args.experiment_name +"                                      & {:.6f}  & {:.6f}       & {:.6f}                         & {:.6f}  & {:.6f}     & {:.6f}  & \multicolumn{{3}}{{c}} {:.6f}           \\ ".format(
                #     fid_giov["FID_giov_/mean"],
                #     (fid_ignite_dict["FID-ignite/ct_mean"]+fid_ignite_dict["FID-ignite/t1_mean"]+fid_ignite_dict["FID-ignite/t2_mean"])/3,
                #     (IS_ignite_dict["IS/ct_mean"]+IS_ignite_dict["IS/t1_mean"]+IS_ignite_dict["IS/t2_mean"])/3,
                #     (dice["DICE/ct"]+dice["DICE/t1"]+dice["DICE/t2"])/3,
                #     (s_score["S-SCORE/ct"]+s_score["S-SCORE/t1"]+s_score["S-SCORE/t2"])/3,
                #     (iou_dict["IoU/ct"]+iou_dict["IoU/t1"]+iou_dict["IoU/t2"])/3,
                #     (mae_dict["mae/ct"]+mae_dict["mae/t1"]+mae_dict["mae/t2"])/3,
                # )
                # wandb.log({"latex_string":formatt},step=ii + 1, commit=True)
            if (i + 1) % 1 == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i + 1, args.epoch)
                print(log)
                torch.cuda.empty_cache()
            if args.preloaded_data:
                idx = idx+1 if (idx+1) <18 else 0
                print("loading dataset ", idx)
                syn_dataset = DroneVeichleDataset(to_be_loaded=True)
                syn_dataset.load_dataset("dataset/tensors",split="train", idx=str(idx), img_size=args.img_size, colored_data=args.color_images)
                syn_loader = DataLoader(syn_dataset, batch_size=args.batch_size, shuffle=True)




def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-datasets', type=str, default='droneveichle')
    parser.add_argument('-experiment_name', type=str, default='testing')
    parser.add_argument('-save_path', type=str, default='s3m-gan-chaos-v1.0')
    
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-img_size', type=int, default=128)
    parser.add_argument('-gan_version', type=str, default='Generator[2/3]+shapeunet+D')
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-sepoch', type=int, default=0)
    parser.add_argument('-preloaded_data', type=bool, default=False)
    parser.add_argument('-color_images', type=bool, default=False)
    parser.add_argument('-modals', type=tuple, )
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-loss_function', type=str, default='wgan-gp+move+cycle+ugan+d+l2')
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-note', type=str,default='affine:True;')
    parser.add_argument('-random_seed', type=int, default='888')
    parser.add_argument('-c_dim', type=int, default='2')
    parser.add_argument('-h_conv', type=int, default='16')
    parser.add_argument('-G_conv', type=int, default='64')
    parser.add_argument('-betas', type=tuple, default=(0.5, 0.9))
    parser.add_argument('-ttur', type=float, default=3e-4)
    parser.add_argument('-w_d_false_c', type=float, default=0.01)
    parser.add_argument('-w_d_false_t_c', type=float, default=0.01)
    parser.add_argument('-w_g_c', type=float, default=1.0)
    parser.add_argument('-w_g_t_c', type=float, default=1.0)
    parser.add_argument('-w_g_cross', type=float, default=50.0)
    parser.add_argument('-w_shape', type=float, default=1)
    parser.add_argument('-w_cycle', type=float, default=1)
    args = parser.parse_args()
    print(args)
    set_deterministic(args.random_seed)
    train(args)