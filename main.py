import argparse
import random
import numpy as np
import torch
from metrics import calculate_all_metrics
from train import train
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
    parser.add_argument('-mode', type=str, default='train')
    parser.add_argument('-dataset', type=str, default='droneveichle')
    parser.add_argument('-dataset_path', type=str, default='dataset')
    parser.add_argument('-experiment_name', type=str, default='testing')
    parser.add_argument('-eval_dir', type=str, default='results')
    parser.add_argument('-save_path', type=str, default='checkpoints')
    parser.add_argument('-save_every', type=int, default=25)
    parser.add_argument('-logs_every', type=int, default=10)
    parser.add_argument('-batch_size', type=int, default=12)
    parser.add_argument('-eval_batch_size', type=int, default=50)
    parser.add_argument('-img_size', type=int, default=128)
    parser.add_argument('-gan_version', type=str, default='Generator[2/3]+shapeunet+D')
    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-sepoch', type=int, default=0)
    parser.add_argument('-preloaded_data', type=bool, default=False)
    parser.add_argument('-pretrained_generator', type=bool, default=False)
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
    
    if args.mode=="train":
        train(args)
    elif args.mode=="eval":
        # eval()
        import models
        net_G = models.Generator(in_c= 3 + args.c_dim, 
                                mid_c=args.G_conv, 
                                layers =2, s_layers=3, 
                                affine=True, last_ac=True,
                                colored_input=True)

        calculate_all_metrics(args, net_G)