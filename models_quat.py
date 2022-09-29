import numpy as np
import torch
from torch import nn

from HLayers.QSN2 import Qspectral_norm
from HLayers.quaternion_layers import QuaternionConv, QuaternionTransposeConv
from HLayers.PHC import PHMConv, PHMTransposeConv

import torch.nn.functional as F
from dataloader import wavelet_wrapper
from torch.nn.utils.parametrizations import spectral_norm
device = "cuda" if torch.cuda.is_available() else "cpu"

class QuaternionInstanceNorm2d(nn.Module):
    r"""Applies a 2D Quaternion Instance Normalization to the incoming data.
        """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super(QuaternionInstanceNorm2d, self).__init__()
        self.num_features = num_features // 4
        self.gamma_init = 1.
        self.affine = affine
        self.gamma = nn.Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.affine)
        self.eps = torch.tensor(1e-5)
        ####
        self.momentum = momentum
        self.track_running_stats = track_running_stats

    def reset_parameters(self):
        self.gamma = nn.Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = nn.Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.affine)

    def forward(self, input):
        # print(self.training)
        quat_components = torch.chunk(input, 4, dim=1)

        r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]

        mu_r = torch.mean(r, axis=(2, 3), keepdims=True)
        mu_i = torch.mean(i, axis=(2, 3), keepdims=True)
        mu_j = torch.mean(j, axis=(2, 3), keepdims=True)
        mu_k = torch.mean(k, axis=(2, 3), keepdims=True)

        mu = torch.stack([torch.mean(mu_r),
                          torch.mean(mu_i),
                          torch.mean(mu_j),
                          torch.mean(mu_k)], dim=0)
        # mu = torch.cat([mu_r,mu_i, mu_j, mu_k], dim=1)

        delta_r, delta_i, delta_j, delta_k = r - mu_r, i - mu_i, j - mu_j, k - mu_k

        quat_variance = torch.mean(delta_r ** 2 + delta_i ** 2 + delta_j ** 2 + delta_k ** 2)
        var = quat_variance

        denominator = torch.sqrt(quat_variance + self.eps)

        # Normalize
        r_normalized = delta_r / denominator
        i_normalized = delta_i / denominator
        j_normalized = delta_j / denominator
        k_normalized = delta_k / denominator

        beta_components = torch.chunk(self.beta, 4, dim=1)

        # Multiply gamma (stretch scale) and add beta (shift scale)
        new_r = (self.gamma * r_normalized) + beta_components[0]
        new_i = (self.gamma * i_normalized) + beta_components[1]
        new_j = (self.gamma * j_normalized) + beta_components[2]
        new_k = (self.gamma * k_normalized) + beta_components[3]

        new_input = torch.cat((new_r, new_i, new_j, new_k), dim=1)
        # if self.track_running_stats:
        #     self.moving_mean.copy_(moving_average_update(self.moving_mean.data, mu.data, self.momentum))
        #     self.moving_var.copy_(moving_average_update(self.moving_var.data, var.data, self.momentum))

        return new_input

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma.shape) \
               + ', beta=' + str(self.beta.shape) \
               + ', eps=' + str(self.eps.shape) + ')'


# def moving_average_update(statistic, curr_value, momentum):
#     term_1 = (1 - momentum) * statistic
#     term_2 = momentum * curr_value
#     new_value = term_1 + term_2
#     return  new_value.data

'''
BLOCKS
'''

bias = False


class conv_block(nn.Module):
    # base block
    def __init__(self, ch_in, ch_out, affine=True, actv=nn.LeakyReLU(inplace=True), downsample=False, upsample=False,
                 share_net_real=False, phm=False, phm_n=4, qsn=False, real=True):
        super(conv_block, self).__init__()
        if phm and not share_net_real:
            self.conv = nn.Sequential(
                PHMConv(phm_n, ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                QuaternionInstanceNorm2d(ch_out, affine=affine),
                actv,
                PHMConv(phm_n, ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                QuaternionInstanceNorm2d(ch_out, affine=affine),
                actv,
            )
        elif qsn and not share_net_real:
            self.conv = nn.Sequential(
                QuaternionConv(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                QuaternionInstanceNorm2d(ch_out, affine=affine),
                actv,
                QuaternionConv(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                QuaternionInstanceNorm2d(ch_out, affine=affine),
                actv
            )
        elif real or share_net_real:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                nn.InstanceNorm2d(ch_out, affine=affine),
                actv,
                nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                nn.InstanceNorm2d(ch_out, affine=affine),
                actv
            )
        self.downsample = downsample
        self.upsample = upsample
        if self.upsample:
            self.up = up_conv(ch_out, ch_out // 2, affine,phm=phm, qsn=qsn, real=real)

    def forward(self, x):
        x1 = self.conv(x)
        c = x1.shape[1]
        if self.downsample:
            x2 = F.avg_pool2d(x1, 2)
            # half of channels for skip
            return x1[:, :c // 2, :, :], x2
        # x1[:,:,:,:]
        if self.upsample:
            x2 = self.up(x1)
            return x2
        return x1


class up_conv(nn.Module):
    # base block
    def __init__(self, ch_in, ch_out, affine=True, actv=nn.LeakyReLU(inplace=True), phm=False, phm_n=4, qsn=False, real=True):
        super(up_conv, self).__init__()

        if phm:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                PHMConv(phm_n, ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                QuaternionInstanceNorm2d(ch_out, affine=affine),
                actv,
            )
        elif qsn:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                QuaternionConv(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                QuaternionInstanceNorm2d(ch_out, affine=affine),
                actv,
            )
        elif real:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=bias),
                nn.InstanceNorm2d(ch_out, affine=affine),
                actv
            )

    def forward(self, x):
        x = self.up(x)
        return x


'''
ENCODER/DECODER
'''


class Encoder(nn.Module):
    # the Encoder_x or Encoder_r of G
    def __init__(self, in_c, mid_c, layers, affine, phm=False, qsn=False, real=True):
        super(Encoder, self).__init__()
        encoder = []
        for i in range(layers):
            encoder.append(conv_block(in_c, mid_c, affine, downsample=True, upsample=False, phm=phm, qsn=qsn, real=real))
            in_c = mid_c
            mid_c = mid_c * 2
        self.encoder = nn.Sequential(*encoder)

    def forward(self, x):
        res = []
        for layer in self.encoder:
            x1, x2 = layer(x)
            res.append([x1, x2])
            x = x2
        return res


class Decoder(nn.Module):
    # the Decoder_x or Decoder_r of G
    def __init__(self, in_c, mid_c, layers, affine, r, phm=False, qsn=False, real=True):
        super(Decoder, self).__init__()
        decoder = []
        for i in range(layers - 1):
            decoder.append(conv_block(in_c - r, mid_c, affine, downsample=False, upsample=True, phm=phm, qsn=qsn, real=real))
            in_c = mid_c
            mid_c = mid_c // 2
            r = r // 2
        decoder.append(conv_block(in_c - r, mid_c, affine, downsample=False, upsample=False, phm=phm, qsn=qsn, real=real))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, share_input, encoder_input):
        encoder_input.reverse()
        x = 0
        for i, layer in enumerate(self.decoder):
            x = torch.cat([share_input, encoder_input[i][0]], dim=1)
            # print(x.shape,share_input.shape, encoder_input[i][0].shape)
            x = layer(x)
            share_input = x
        return x


'''
SHARE NET
'''


class ShareNet(nn.Module):
    # the Share Block of G
    def __init__(self, in_c, out_c, layers, affine, r, share_net_real=True, phm=False, qsn=False, real=True):
        super(ShareNet, self).__init__()
        encoder = []
        decoder = []
        for i in range(layers - 1):
            encoder.append(conv_block(in_c, in_c * 2, affine, downsample=True, upsample=False,
                                      share_net_real=share_net_real, phm=phm, qsn=qsn, real=real))
            decoder.append(conv_block(out_c - r, out_c // 2, affine, downsample=False, upsample=True,
                                      share_net_real=share_net_real, phm=phm, qsn=qsn, real=real))
            in_c = in_c * 2
            out_c = out_c // 2
            r = r // 2
        self.bottom = conv_block(in_c, in_c * 2, affine, upsample=True, phm=phm, qsn=qsn, real=real)
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)
        self.layers = layers

    def forward(self, x):
        encoder_output = []
        x = x[-1][1]
        for layer in self.encoder:
            x1, x2 = layer(x)
            encoder_output.append([x1, x2])
            x = x2
        bottom_output = self.bottom(x)
        if self.layers == 1:
            return bottom_output
        encoder_output.reverse()
        for i, layer in enumerate(self.decoder):
            x = torch.cat([bottom_output, encoder_output[i][0]], dim=1)
            x = layer(x)
            bottom_output = x
        return x


'''
DISCRIMINATOR
'''


class Discriminator(nn.Module):
    # the D_x or D_r of TarGAN ( backbone of PatchGAN )

    def __init__(self, image_size=256, conv_dim=64, c_dim=5, repeat_num=6, colored_input=True, real=True, qsn=False, phm=False, phm_n=4, spectral=False, last_layer_gen_real=True):
        super(Discriminator, self).__init__()
        layers = []
        if phm:
            layers.append(PHMConv(phm_n, phm_n, conv_dim, kernel_size=4, stride=2, padding=1))
        elif qsn:
            if spectral:
                layers.append(Qspectral_norm(QuaternionConv(4, conv_dim, kernel_size=4, stride=2, padding=1)))
            else:
                layers.append(QuaternionConv(4, conv_dim, kernel_size=4, stride=2, padding=1))
            # layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        elif real:
            if spectral:
                layers.append(spectral_norm(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1)))
            else:
                layers.append(nn.Conv2d(1 if not colored_input else 3, conv_dim, kernel_size=4, stride=2, padding=1))

        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            if phm:
                layers.append(PHMConv(phm_n, curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            elif qsn:
                if spectral:
                    layers.append(
                        Qspectral_norm(QuaternionConv(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
                else:
                    layers.append(QuaternionConv(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            elif real:
                if spectral:
                    layers.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1)))
                else:
                    layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))

            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)

        if phm:
            self.conv1 = PHMConv(phm_n, curr_dim, 4, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = PHMConv(phm_n, curr_dim, c_dim, kernel_size=kernel_size, bias=False)
            # self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
            # print(curr_dim,c_dim,kernel_size)

        elif qsn:
            # self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
            # self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
            if spectral:
                self.conv1 = Qspectral_norm(
                    QuaternionConv(in_channels=curr_dim, out_channels=4, kernel_size=3, stride=1, padding=1,
                                   bias=False))
                self.conv2 = Qspectral_norm(
                    QuaternionConv(in_channels=curr_dim, out_channels=c_dim, kernel_size=kernel_size, stride=1,
                                   bias=False))
            else:
                self.conv1 = QuaternionConv(in_channels=curr_dim, out_channels=4, kernel_size=3, stride=1, padding=1,
                                            bias=False)
                self.conv2 = QuaternionConv(in_channels=curr_dim, out_channels=c_dim, kernel_size=kernel_size, stride=1,
                                            bias=False)
        elif real:
            if spectral:
                self.conv1 = spectral_norm(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
                self.conv2 = spectral_norm(nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False))
            else:
                self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
                self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

        if last_layer_gen_real:
            if spectral:
                self.conv1 = spectral_norm(nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False))
                self.conv2 = spectral_norm(nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False))
            else:
                self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
                self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        self.real = real
        self.last_layer_gen_real = last_layer_gen_real
        self.qsn = qsn
        self.phm = phm
        self.in_c = 1 if not colored_input else 3 if real else 4
    def forward(self, x_real):
        # print("discriminatore in entrata", x.shape) #torch.Size([4, 1, 128, 128])
        # rgb + aalpha
        # FOR WAVELET COMMENTED
        # if (not args.real and args.soup) and not args.last_layer_gen_real:
        #     x = x.repeat(1, 3, 1, 1)
        #     x = torch.cat([x, grayscale(x)], 1)
        # print("discriminatore dopo main",h.shape) #torch.Size([4, 2048, 2, 2])
        if self.qsn or self.phm:
            x_real = torch.cat((x_real, torch.zeros(x_real.shape[0], self.in_c-x_real.shape[1] ,x_real.shape[2],x_real.shape[3]).to(device)), dim=1).to(device)
        h = self.main(x_real)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)

        # print("discriminatore out src",out_src.shape) #torch.Size([4, 1, 2, 2])
        # print("discriminatore out cls",out_cls.shape, "view",out_cls.view(out_cls.size(0), out_cls.size(1)).shape) #torch.Size([4, 6, 1, 1]) view torch.Size([4, 6])
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))



'''
Generator
'''
@torch.no_grad()
def create_wavelet_from_input_tensor(inputs, mods, wav_type ):
    modalities = ["ir" if mods[i][0].any()==1 else "rgb" if mods[i][1].any()==1 else "" for i in range(mods.size(0))]
    # .permute(1, 2,0)
    lst = [
        torch.from_numpy(wavelet_wrapper(wav_type, chunk.squeeze().cpu().detach(), chunk.size(2), modalities[i])).type(torch.FloatTensor)
        for i,chunk in enumerate(torch.split(inputs.detach(), 1, dim=0))
        ]
    return torch.stack(lst, dim=0).to(device)



class Generator(nn.Module):
    # the G of TarGAN

    def __init__(self, in_c, mid_c, layers, s_layers, affine, last_ac=True, colored_input=True, wav=False, real=True, qsn=False, phm=False, phm_n=4, spectral=False, last_layer_gen_real=True):
        super(Generator, self).__init__()
        self.img_encoder = Encoder(in_c, mid_c, layers, affine, phm=phm, qsn=qsn, real=real)
        self.img_decoder = Decoder(mid_c * (2 ** layers), mid_c * (2 ** (layers - 1)), layers, affine,64)
        self.in_c_targ = in_c-4 if wav is not None else in_c
        self.target_encoder = Encoder(self.in_c_targ, mid_c, layers, affine, phm=phm, qsn=qsn, real=real)
        self.target_decoder = Decoder(mid_c * (2 ** layers), mid_c * (2 ** (layers - 1)), layers, affine,64) 
        self.share_net = ShareNet(mid_c * (2 ** (layers - 1)), mid_c * (2 ** (layers - 1 + s_layers)), s_layers, affine,256)

        if phm and not last_layer_gen_real:
            self.out_img = PHMConv(phm_n, mid_c, 4, 1, bias=bias)
            self.out_tumor = PHMConv(phm_n, mid_c, 4, 1, bias=bias)
        elif qsn and not last_layer_gen_real:
            self.out_img = QuaternionConv(mid_c, 4, 1, stride=1, bias=bias)
            self.out_tumor = QuaternionConv(mid_c, 4, 1, stride=1, bias=bias)
        elif real or last_layer_gen_real:
            self.out_img = nn.Conv2d(mid_c, 1 if not colored_input else 3, 1, bias=bias)
            self.out_tumor = nn.Conv2d(mid_c, 1 if not colored_input else 3, 1, bias=bias)

        self.last_ac = last_ac
        self.real = real
        self.qsn = qsn
        self.in_c = in_c
        self.phm = phm
    # G(image,target_image,target_modality) --> (out_image,output_target_area_image)

    def forward(self, img, tumor=None, c=None, mode="train", wav_type=None):
        # print("input img shape",img.shape, c.shape) torch.Size([4, 1, 128, 128]) torch.Size([4, 3])
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, img.size(2), img.size(3))
        if wav_type != None:
            img = torch.cat([img, create_wavelet_from_input_tensor(img, c, wav_type)], dim=1)
        
        img = torch.cat([img, c], dim=1)
        if self.qsn or self.phm:
            img = torch.cat((img, torch.zeros(img.shape[0], self.in_c-img.shape[1] ,img.shape[2],img.shape[3]).to(device)), dim=1).to(device)
        x_1 = self.img_encoder(img)
        s_1 = self.share_net(x_1)
        res_img = self.out_img(self.img_decoder(s_1,x_1))

        # print(res_img.shape)#torch.Size([4, 1, 128, 128])
        if self.last_ac:
            res_img = torch.tanh(res_img)
        if mode == "train":
            tumor = torch.cat([tumor, c], dim=1)
            if self.qsn or self.phm:
                tumor = torch.cat((tumor, torch.zeros(tumor.shape[0], self.in_c_targ-tumor.shape[1] ,tumor.shape[2],tumor.shape[3]).to(device)), dim=1).to(device)
            x_2 = self.target_encoder(tumor)
            s_2 = self.share_net(x_2)
            res_tumor = self.out_tumor(self.target_decoder(s_2, x_2))
            if self.last_ac:
                res_tumor = torch.tanh(res_tumor)


            return res_img, res_tumor
        return res_img


'''
SHAPE U NET
'''


class ShapeUNet(nn.Module):
    # the S of TarGAN

    def __init__(self, img_ch=1, mid=32, output_ch=1, real=True, qsn=False, phm=False, phm_n=4, last_layer_gen_real=True):
        super(ShapeUNet, self).__init__()
        self.img_ch = img_ch
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=mid)

        
        self.Conv2 = conv_block(ch_in=mid, ch_out=mid * 2)
        self.Conv3 = conv_block(ch_in=mid * 2, ch_out=mid * 4)
        self.Conv4 = conv_block(ch_in=mid * 4, ch_out=mid * 8)
        self.Conv5 = conv_block(ch_in=mid * 8, ch_out=mid * 16)

        if phm:
            self.Up5 = PHMTransposeConv(phm_n, mid * 16, mid * 8, kernel_size=2, stride=2)
            self.Up4 = PHMTransposeConv(phm_n, mid * 8, mid * 4, kernel_size=2, stride=2)
            self.Up3 = PHMTransposeConv(phm_n, mid * 4, mid * 2, kernel_size=2, stride=2)
            self.Up2 = PHMTransposeConv(phm_n, mid * 2, mid * 1, kernel_size=2, stride=2)
            self.Conv_1x1 = PHMConv(phm_n, mid * 1, output_ch, kernel_size=1)
        elif qsn:
            self.Up5 = QuaternionTransposeConv(mid * 16, mid * 8, kernel_size=2, stride=2)
            self.Up4 = QuaternionTransposeConv(mid * 8, mid * 4, kernel_size=2, stride=2)
            self.Up3 = QuaternionTransposeConv(mid * 4, mid * 2, kernel_size=2, stride=2)
            self.Up2 = QuaternionTransposeConv(mid * 2, mid * 1, kernel_size=2, stride=2)
            self.Conv_1x1 = QuaternionConv(mid * 1, output_ch, kernel_size=1, stride=1)
            # self.Conv_1x1 = nn.Conv2d(mid * 1, output_ch, kernel_size=1)

        elif real:
            self.Up5 = nn.ConvTranspose2d(mid * 16, mid * 8, kernel_size=2, stride=2)
            self.Up4 = nn.ConvTranspose2d(mid * 8, mid * 4, kernel_size=2, stride=2)
            self.Up3 = nn.ConvTranspose2d(mid * 4, mid * 2, kernel_size=2, stride=2)
            self.Up2 = nn.ConvTranspose2d(mid * 2, mid * 1, kernel_size=2, stride=2)
            self.Conv_1x1 = nn.Conv2d(mid * 1, output_ch, kernel_size=1)
        if last_layer_gen_real:
            self.Conv_1x1 = nn.Conv2d(mid * 1, output_ch, kernel_size=1)

        self.Up_conv5 = conv_block(ch_in=mid * 16, ch_out=mid * 8)
        self.Up_conv4 = conv_block(ch_in=mid * 8, ch_out=mid * 4)
        self.Up_conv3 = conv_block(ch_in=mid * 4, ch_out=mid * 2)
        self.Up_conv2 = conv_block(ch_in=mid * 2, ch_out=mid * 1)
        self.phm = phm
        self.qsn = qsn
    def forward(self, x):
        # encoding path
        # if (not args.real and args.soup) and not args.last_layer_gen_real:
        #     x = x.repeat(1, 3, 1, 1)
        #     x = torch.cat([x, grayscale(x)], 1)
        # wavelet
        if self.qsn or self.phm:
            x = torch.cat((x, torch.zeros(x.shape[0], self.img_ch-x.shape[1] ,x.shape[2],x.shape[3]).to(device)), dim=1).to(device)
      
        x1 = self.Conv1(x)

        x2 = self.Maxpool1(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool2(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool3(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool4(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)


        return torch.sigmoid(d1)