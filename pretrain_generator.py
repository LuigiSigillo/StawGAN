import torch
from torch import nn
from dataloader import DroneVeichleDatasetPreTraining
from models import Generator
from tqdm import tqdm
import utils
from torch.utils.data import DataLoader

def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    for e in range(epochs):
        loss_meter = utils.AverageMeter()
        for data in tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_meter.update(loss.item(), L.size(0))
            
        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")

device = "cuda"
in_c = 3
net_G = Generator(in_c = in_c + 2,
                  mid_c = 64,
                  layers = 2,
                  s_layers=3,
                  affine=True,
                  last_ac=True,
                  colored_input=True)
opt = torch.optim.Adam(net_G.parameters(), lr=1e-4)
criterion = nn.L1Loss()
data = DroneVeichleDatasetPreTraining(split="train")
pretrain_generator(net_G, DataLoader(data), opt, criterion, 20)

torch.save(net_G.state_dict(), "pretrained_gen.pt")