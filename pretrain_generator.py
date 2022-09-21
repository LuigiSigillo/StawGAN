import torch
from torch import nn
from dataloader import DroneVeichleDatasetPreTraining
from models import Generator
from tqdm import tqdm
import utils 
from torch.utils.data import DataLoader

def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    for e in range(epochs):
        for data in tqdm(train_dl):
            ir, rgb, lab = data[0].to(device), data[1].to(device), data[2].to(device)
            c_trg = ut.label2onehot(lab, 2)

            preds = net_G(ir, c_trg=c_trg, mode="pretrain")
            loss = criterion(preds, rgb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # loss_meter.update(loss.item(), L.size(0))
            
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