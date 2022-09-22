import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
from pathlib import Path
import wandb
import torchvision.utils as vutils


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def save_image(x, ncol, filename,args):
    x = denormalize(x)

    # IF BATCH
    if len(x.shape) == 4:
        if x.size(1) == 3: # non quat
            x = x
        else: #quat
            #x = x[:,1:4,:,:]
            x = x[:,0:3,:,:]
    
        sample_dir = " ".join(filename.replace(args.experiment_name,"").replace(".jpg","").split("_")[1:])
        if args.mode=="train":
            iters = str(int(filename.replace(args.experiment_name,"").split("/")[3].split("_")[0]))

    # IS ONLY ONE PHOTO
    else:
        if x.size(0) == 3: # non quat
            x = x
        else: #quat
            #x = x[:,1:4,:,:]
            x = x[0:3,:,:]
        #iters = str(int(filename.split("/")[9].split("_")[0]))
    
    vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)
    if len(x.shape) == 4 and args.mode=="train":
        wandb.log({sample_dir: wandb.Image(filename,caption=iters)},commit=False)

def loss_filter(mask,device="cuda"):
    list = []
    for i, m in enumerate(mask):
        if torch.any(m == 1):
            list.append(i)
    index = torch.tensor(list, dtype=torch.long).to(device)
    return index

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

def denorm(x):
    res = (x + 1.) / 2.
    res.clamp_(0, 1)
    return res

def renorm(x):
    res = (x - 0.5) / 0.5
    res.clamp_(-1, 1)
    return res


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    for i in range(batch_size):
        out[i, labels[i].long()] = 1
    return out

def getLabel(imgs, device, index, c_dim=2):
    syn_labels = torch.zeros((imgs.size(0), c_dim)).to(device)
    syn_labels[:, index] = 1.
    return syn_labels

def gradient_penalty(y, x, device):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]
    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    return torch.mean((dydx_l2norm - 1) ** 2)


def save_state_net(net, args, index, optim=None, experiment_name="test"):
    save_path = os.path.join(args.save_path, args.experiment_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_file = os.path.join(save_path, args.net_name)
    torch.save(net.state_dict(), save_file + '_' + str(index) +"_"+experiment_name + '.pkl')
    if optim is not None:
        torch.save(optim.state_dict(), save_file + '_optim_' + str(index)+ "_" + experiment_name + '.pkl')
    if not os.path.isfile(save_path + '/outputs.txt'):
        with open(save_path + '/outputs.txt', mode='w') as f:
            argsDict = args.__dict__;
            f.writelines(args.note + '\n')
            for i in argsDict.keys():
                f.writelines(str(i) + ' : ' + str(argsDict[i]) + '\n')

def load_state_net(net, net_name, index, optim=None, experiment_name="", device="cpu"):
    save_path = os.path.join(save_path, experiment_name)
    if not os.path.isdir(save_path):
        raise Exception("wrong path")
    save_file = os.path.join(save_path, net_name)
    if net is not None:
        net.load_state_dict(torch.load(save_file + '_' + str(index)+ "_"  + experiment_name + '.pkl', map_location=device))
    if optim is not None:
        optim.load_state_dict(torch.load(save_file + '_optim_' + str(index)+ "_"  + experiment_name + '.pkl'))
    return net, optim

def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)



def plot_images(netG_use, syneval_dataset, device, c_dim):
    # fig = plt.figure(dpi=120)
    idx = random.randint(0,200)
    with torch.no_grad():
        img = syneval_dataset[idx][0]
        trg_segm = syneval_dataset[idx][3]
        img = img.unsqueeze(dim=0).to(device)
        trg_orig = syneval_dataset[idx][1]
        trg_orig = trg_orig.unsqueeze(dim=0).to(device)

        # print(getLabel(img, device, 0, args.c_dim).shape,img.shape)
        pred_t1_img, pred_t1_targ = netG_use(img, trg_orig, c=getLabel(img, device, 0, c_dim), )
        pred_t2_img, pred_t2_targ = netG_use(img, trg_orig, c=getLabel(img, device, 1, c_dim), )
        # plt.subplot(241)
        # plt.imshow(denorm(img).squeeze().cpu().numpy())
        # plt.title(str(i + 1) + '_source')
        # plt.subplot(242)
        # plt.imshow(denorm(pred_t1_img).squeeze().cpu().numpy(), )
        # plt.title('pred_x1')
        # plt.subplot(243)
        # plt.imshow(denorm(pred_t2_img).squeeze().cpu().numpy(), cmap='gray')
        # plt.title('pred_x2')
        # plt.subplot(244)
        # plt.imshow(denorm(pred_t3_img).squeeze().cpu().numpy(), cmap='gray')
        # plt.title('pred_x3')
        # plt.show()
        # x_concat = []
        # x_concat = torch.cat(x_concat, dim=0)
        # plt.close(fig)
    return denorm(img).cpu(), \
            denorm(pred_t1_img).cpu(), \
            denorm(pred_t2_img).cpu(), \
            denorm(trg_orig).cpu(), \
            pred_t1_targ.cpu(), \
            pred_t2_targ.cpu()


def load_nets(nets,sepoch):
    for net in nets.keys():
        print("loading", net)
        net_check = net if "use" in net else net.replace("_", "")
        load_state_net(nets[net], net_check, sepoch)

def listdir(dname):
    fnames = list(chain(*[list(Path(dname).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames