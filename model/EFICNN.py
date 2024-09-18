import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# -------------------------------------------------Iterative VGG16-----------------------------------------------------#
class ASPP(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        kernel_sizes = [1, 3, 3, 1]
        dilations = [1, 3, 6, 1]
        paddings = [0, 3, 6, 0]
        self.aspp = torch.nn.ModuleList()
        for aspp_idx in range(len(kernel_sizes)):
            conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_sizes[aspp_idx],
                stride=1,
                dilation=dilations[aspp_idx],
                padding=paddings[aspp_idx],
                bias=True)
            self.aspp.append(conv)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.aspp_num = len(kernel_sizes)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(self.aspp_num):
            inp = avg_x if (aspp_idx == self.aspp_num - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out


class Iter_VGG16(nn.Module):
    def __init__(self):
        super(Iter_VGG16, self).__init__()
        model = models.vgg16()

        features = list(model.features)[:24]
        self.features = nn.ModuleList(features).eval()
        self.net1 = nn.Sequential(*features[:5])
        self.net2 = nn.Sequential(*features[5:10])
        self.net3 = nn.Sequential(*features[10:17])
        self.net4 = nn.Sequential(*features[17:24])

        self.aspp1 = ASPP(128, 16)
        self.aspp2 = ASPP(256, 32)
        self.aspp3 = ASPP(512, 64)
        self.aspp4 = ASPP(1024, 128)

    def forward(self, x):
        a1 = self.net1(x)
        b1 = self.net2(a1)
        c1 = self.net3(b1)
        d1 = self.net4(c1)

        a2 = self.net1(x)
        a12 = self.aspp1(torch.cat([a1, a2], 1))
        b2 = self.net2(a12)
        b12 = self.aspp2(torch.cat([b1, b2], 1))
        c2 = self.net3(b12)
        c12 = self.aspp3(torch.cat([c1, c2], 1))
        d2 = self.net4(c12)
        d12 = self.aspp4(torch.cat([d1, d2], 1))
        return a12, b12, c12, d12


# ------------------------------------------------------3D-DEM---------------------------------------------------------#
class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        att_weight = self.activaton(y)
        return att_weight


class ThreeD_DEM(nn.Module):
    def __init__(self, in_channels):
        super(ThreeD_DEM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.att1 = SimAM()
        self.att2 = SimAM()

    def forward(self, x1, x2):
        diff = torch.abs(x1 - x2)
        diff_att = self.att1.forward(diff)
        diff = diff * diff_att

        con1 = x1 * diff_att + x1
        con2 = x2 * diff_att + x2

        cat = torch.cat([con1, con2], dim=1)
        cat = self.conv(cat)
        cat_att = self.att2.forward(cat)
        cat = cat_att * cat

        fuse_feature = diff + cat
        return fuse_feature


# -------------------------------------------------------EGAM----------------------------------------------------------#
class Laplacian:
    def __init__(self, channels=3, cuda=True):
        self.channels = channels
        self.cuda = cuda
        self.kernel = self.create_gauss_kernel()

    def create_gauss_kernel(self):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        if self.cuda:
            kernel = kernel.cuda()
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def conv_gauss(self, img):
        img = F.pad(img, (2, 2, 2, 2), mode='reflect')
        out = F.conv2d(img, self.kernel, groups=img.shape[1])
        return out

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up)

    @staticmethod
    def make_laplace(img, channels):
        laplacian = Laplacian(channels=channels)
        filtered = laplacian.conv_gauss(img)
        down = laplacian.downsample(filtered)
        up = laplacian.upsample(down)
        if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
            up = F.interpolate(up, size=(img.shape[2], img.shape[3]), mode="bilinear", align_corners=True)
        diff = img - up
        return diff


class EGAM(nn.Module):
    def __init__(self, in_channels):
        super(EGAM, self).__init__()

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.fuse_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())

        self.edge_out = nn.Conv2d(in_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.laplacian = Laplacian()

    def forward(self, x):
        residual = x
        edge = self.edge_out(x)
        edge_att = self.sigmoid(edge)

        # background attention
        background_att = 1 - edge_att
        background_x = x * background_att

        # high-frequency edge attention
        edge_att = self.laplacian.make_laplace(edge_att, 1)
        edge_x = x * edge_att

        fusion_x = torch.cat([background_x, edge_x], dim=1)
        fusion_x = self.fusion_conv(fusion_x)

        attention_map = self.fuse_attention(fusion_x)
        fuse_feature = fusion_x * attention_map + residual
        return fuse_feature


# -------------------------------------------------------FGFM----------------------------------------------------------#
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class FGFM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FGFM, self).__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.flow_make = nn.Conv2d(out_channels * 2, 4, 3, padding=1, bias=False)
        self.flow_gate = nn.Sequential(
            nn.Conv2d(4, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.ca = ChannelAttentionModule(out_channels, ratio=16)

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output

    def forward(self, lowres_feature, highres_feature):
        h_feature = highres_feature
        h, w = highres_feature.size()[2:]
        size = (h, w)

        l_feature = self.down(lowres_feature)
        l_feature_up = F.interpolate(l_feature, size=size, mode="bilinear", align_corners=True)

        flow = self.flow_make(torch.cat([l_feature_up, h_feature], 1))
        flow_l, flow_h = flow[:, :2, :, :], flow[:, 2:, :, :]

        l_feature_warp = self.flow_warp(l_feature, flow_l, size=size)
        h_feature_warp = self.flow_warp(h_feature, flow_h, size=size)

        feature_cat = l_feature_warp + h_feature_warp
        flow_gates = self.ca(feature_cat)

        fuse_feature = l_feature_warp * flow_gates + h_feature_warp * (1 - flow_gates)
        return fuse_feature

# ------------------------------------------------------EFICNN---------------------------------------------------------#
class ConvModule(nn.Module):
    def __init__(self, in_channels):
        super(ConvModule, self).__init__()
        self.channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x_out = self.conv(x)
        return x_out


class EFICNN(nn.Module):
    def __init__(self):
        super(EFICNN, self).__init__()
        self.backbone = Iter_VGG16()

        self.dem1 = ThreeD_DEM(64)
        self.dem2 = ThreeD_DEM(128)
        self.dem3 = ThreeD_DEM(256)
        self.dem4 = ThreeD_DEM(512)

        self.egam1 = EGAM(64)
        self.egam2 = EGAM(128)
        self.egam3 = EGAM(256)
        self.egam4 = EGAM(512)

        self.conv1 = ConvModule(64)
        self.conv2 = ConvModule(128)
        self.conv3 = ConvModule(256)
        self.conv4 = ConvModule(512)

        self.fgfm43 = FGFM(512, 256)
        self.fgfm32 = FGFM(256, 128)
        self.fgfm21 = FGFM(128, 64)
        self.fgfm31 = FGFM(256, 64)
        self.fgfm41 = FGFM(512, 64)

        self.convout = nn.Conv2d(64, 2, 1)

    def forward(self, t1_input, t2_input):
        t1_list = self.backbone(t1_input)
        t2_list = self.backbone(t2_input)

        t1_x1, t1_x2, t1_x3, t1_x4 = t1_list[0], t1_list[1], t1_list[2], t1_list[3]
        t2_x1, t2_x2, t2_x3, t2_x4 = t2_list[0], t2_list[1], t2_list[2], t2_list[3]

        f1 = self.dem1(t1_x1, t2_x1)
        f2 = self.dem2(t1_x2, t2_x2)
        f3 = self.dem3(t1_x3, t2_x3)
        f4 = self.dem4(t1_x4, t2_x4)

        fe1 = self.egam1(f1)
        fe2 = self.egam2(f2)
        fe3 = self.egam3(f3)
        fe4 = self.egam4(f4)

        ff4 = self.conv4(fe4)
        ff3 = self.fgfm43(ff4, fe3)
        ff3 = self.conv3(ff3)

        ff2 = self.fgfm32(ff3, fe2)
        ff2 = self.conv2(ff2)

        ff1 = self.fgfm21(ff2, fe1)
        ff1 = self.conv1(ff1)

        ff2_up = self.fgfm21(ff2, ff1)
        ff3_up = self.fgfm31(ff3, ff1)
        ff4_up = self.fgfm41(ff4, ff1)
        ff = ff1 + ff2_up + ff3_up + ff4_up

        ff_up = F.interpolate(ff, scale_factor=2, mode="bilinear", align_corners=True)
        out = self.convout(ff_up)
        return (out, )
