import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np

# Complete code
# -------------------------------------------------Iterative VGG16-----------------------------------------------------#
class Up_add(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up_add, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1)

    def forward(self, x1, x2):
        x1_conv = self.conv(x1)
        size = x2.size()[2:]
        x1_up = F.interpolate(x1_conv, size=size, mode='bilinear')
        out = x1_up + x2

        return out


class Iter_VGG16(nn.Module):
    def __init__(self):
        super(Iter_VGG16, self).__init__()
        model = models.vgg16(pretrained=False)
        pre = torch.load("model/vgg16-397923af.pth")
        model.load_state_dict(pre)

        self.net1 = model.features[:5]
        self.net2 = model.features[5:10]
        self.net3 = model.features[10:17]
        self.net4 = model.features[17:24]

        self.up_add4 = Up_add(512, 256)
        self.up_add3 = Up_add(256, 128)
        self.up_add2 = Up_add(128, 64)

        self.smooth1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.smooth2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.smooth4 = nn.Conv2d(512, 512, 3, 1, 1)

        self.conv1 = nn.Conv2d(64, 1, 1, 1)
        self.conv2 = nn.Conv2d(128, 1, 1, 1)
        self.conv3 = nn.Conv2d(256, 1, 1, 1)
        self.conv4 = nn.Conv2d(512, 1, 1, 1)

    def forward(self, x):
        # First Backbone
        a1 = self.net1(x)
        a2 = self.net2(a1)
        a3 = self.net3(a2)
        a4 = self.net4(a3)

        a3 = self.up_add4(a4, a3)
        a2 = self.up_add3(a3, a2)
        a1 = self.up_add2(a2, a1)

        a1 = self.smooth1(a1)
        a2 = self.smooth2(a2)
        a3 = self.smooth3(a3)
        a4 = self.smooth4(a4)

        # Second Backbone
        b1 = self.net1(x)
        b2 = self.net2(a1 + b1)
        b3 = self.net3(a2 + b2)
        b4 = self.net4(a3 + b3)

        b3 = self.up_add4(b4, b3)
        b2 = self.up_add3(b3, b2)
        b1 = self.up_add2(b2, b1)

        b1 = self.smooth1(b1)
        b2 = self.smooth2(b2)
        b3 = self.smooth3(b3)
        b4 = self.smooth4(b4)

        w1 = torch.sigmoid(self.conv1(b1))
        b1 = b1 * w1 + a1 * (1 - w1)

        w2 = torch.sigmoid(self.conv2(b2))
        b2 = b2 * w2 + a2 * (1 - w2)

        w3 = torch.sigmoid(self.conv3(b3))
        b3 = b3 * w3 + a3 * (1 - w3)

        w4 = torch.sigmoid(self.conv4(b4))
        b4 = b4 * w4 + a4 * (1 - w4)

        return b1, b2, b3, b4


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
class LaplacianExtractor(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须为奇数")

        radius = kernel_size // 2
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")

        gaussian = torch.exp(-(grid_x.pow(2) + grid_y.pow(2)) / (2.0 * sigma ** 2))
        gaussian = gaussian / gaussian.sum()

        self.padding = radius
        self.register_buffer("blur_weight", gaussian.unsqueeze(0).unsqueeze(0))

    def blur(self, feature):
        channel_num = feature.shape[1]
        weight = self.blur_weight.to(
            device=feature.device,
            dtype=feature.dtype
        ).expand(channel_num, 1, -1, -1)

        padded = F.pad(feature, (self.padding,) * 4, mode="reflect")
        return F.conv2d(padded, weight, groups=channel_num)

    def forward(self, feature):
        smooth_feature = self.blur(feature)
        reduced_feature = F.avg_pool2d(smooth_feature, kernel_size=2, stride=2)
        reconstructed_feature = F.interpolate(
            reduced_feature,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        reconstructed_feature = self.blur(reconstructed_feature)
        return feature - reconstructed_feature


class EGAM(nn.Module):
    def __init__(self, in_channels, kernel_size=5, sigma=1.0):
        super().__init__()

        self.edge_predictor = nn.Conv2d(in_channels, 1, kernel_size=1, bias=True)
        self.frequency_extractor = LaplacianExtractor(
            kernel_size=kernel_size,
            sigma=sigma
        )

        self.feature_mixer = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, feature):
        identity = feature

        edge_probability = self.edge_predictor(feature).sigmoid()
        context_feature = feature * (1.0 - edge_probability)

        edge_detail = self.frequency_extractor(edge_probability)
        boundary_feature = feature * edge_detail

        mixed_feature = torch.cat((context_feature, boundary_feature), dim=1)
        mixed_feature = self.feature_mixer(mixed_feature)

        gate = self.spatial_gate(mixed_feature)
        enhanced_feature = mixed_feature * gate

        return identity + enhanced_feature


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
        out = self.conv(x)
        return out


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

        self.out = nn.Conv2d(64, 1, kernel_size=1)

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

        ff = ff1 + ff2_up + ff3_up + ff4_up  # 128 128 64

        # Change map
        ff_out = F.interpolate(self.out(ff), scale_factor=(2, 2), mode='bilinear')
        ff_out = torch.sigmoid(ff_out)

        ff2_out = F.interpolate(self.out(ff2_up), scale_factor=(2, 2), mode='bilinear')
        ff2_out = torch.sigmoid(ff2_out)

        ff3_out = F.interpolate(self.out(ff3_up), scale_factor=(2, 2), mode='bilinear')
        ff3_out = torch.sigmoid(ff3_out)

        ff4_out = F.interpolate(self.out(ff4_up), scale_factor=(2, 2), mode='bilinear')
        ff4_out = torch.sigmoid(ff4_out)

        return ff_out, ff2_out, ff3_out, ff4_out


# if __name__ == '__main__':
#     from torchstat import stat
#     model = EFICNN()
#     stat(model, (3, 256, 256))

# ===================================================================================================================================================================
# Total params: 21,324,821
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Total memory: 234.71MB
# Total MAdd: 57.95GMAdd
# Total Flops: 29.02GFlops
# Total MemR+W: 595.72MB
