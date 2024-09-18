import math
import torch
import torch.nn.functional as F
from torch import nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out


class BAM(nn.Module):
    """ Basic self-attention module
    """

    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(BAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in //8
        self.activation = activation
        self.ds = ds  #
        self.pool = nn.AvgPool2d(self.ds)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, input):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        x = self.pool(input)
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel**-.5) * energy

        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = F.interpolate(out, [width*self.ds,height*self.ds])
        out = out + input

        return out


class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=1,stride=1,padding=0,groups=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True))

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class AERNet(nn.Module):
    def __init__(self, freeze_bn=False):
        super(AERNet, self).__init__()
        self.encoder = resnet34()  # 在此处可切换backbone
        self.decoder = Decoder()
        self.classifier = nn.Conv2d(1, 2, 1, padding=0, bias=False)
        if freeze_bn:
            self.freeze_bn()

    def forward(self, A, B):
        output1 = self.encoder(A)
        output2 = self.encoder(B)
        result = self.decoder(output1, output2)
        result = self.classifier(result)
        return result

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', }


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return (out,)


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        feature.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        feature.append(x)
        x = self.layer2(x)
        feature.append(x)
        x = self.layer3(x)
        feature.append(x)
        x = self.layer4(x)
        feature.append(x)
        return feature


def resnet34(pretrained=True):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        pretrained_path = "model/backbone/resnet34-333f7ec4.pth"
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict, strict=False)
    return model


class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_block, self).__init__()

        self.de_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        self.de_block2 = DWConv(out_channels, out_channels)

        self.att = CoordAtt(out_channels, out_channels)

        self.de_block3 = DWConv(out_channels, out_channels)

        self.de_block4 = nn.Conv2d(out_channels, 1, 1)

        self.de_block5 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, input1, input, input2):
        x0 = torch.cat((input1, input, input2), dim=1)
        x0 = self.de_block1(x0)
        x = self.de_block2(x0)
        x = self.att(x)
        x = self.de_block3(x)
        x = x + x0
        al = self.de_block4(x)
        result = self.de_block5(x)

        return al, result


class ref_seg(nn.Module):
    def __init__(self):
        super(ref_seg, self).__init__()
        self.dir_head = nn.Sequential(nn.Conv2d(32, 32, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 8, 1, 1))
        self.conv0 = nn.Conv2d(1, 8, 3, 1, 1, bias=False)
        self.conv0.weight = nn.Parameter(torch.tensor([[[[0, 0, 0], [1, 0, 0], [0, 0, 0]]],
                                                       [[[1, 0, 0], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0, 1, 0], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0, 0, 1], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0, 0, 0], [0, 0, 1], [0, 0, 0]]],
                                                       [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]],
                                                       [[[0, 0, 0], [0, 0, 0], [0, 1, 0]]],
                                                       [[[0, 0, 0], [0, 0, 0], [1, 0, 0]]]]).float())

    def forward(self, x, masks_pred, edge_pred):
        direc_pred = self.dir_head(x)
        direc_pred = direc_pred.softmax(1)
        edge_mask = 1 * (torch.sigmoid(edge_pred).detach() > 0.5)
        refined_mask_pred = (self.conv0(masks_pred) * direc_pred).sum(1).unsqueeze(1) * edge_mask + masks_pred * (
                    1 - edge_mask)
        return refined_mask_pred


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.bam = BAM(1024)
        self.db1 = nn.Sequential(
            nn.Conv2d(1024, 512, 1), nn.BatchNorm2d(512), nn.ReLU(),
            DWConv(512, 512),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        )

        self.db2 = decoder_block(1024, 256)
        self.db3 = decoder_block(512, 128)
        self.db4 = decoder_block(256, 64)
        self.db5 = decoder_block(192, 32)

        self.classifier1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, 1))

        self.classifier2 = nn.Sequential(
            nn.Conv2d(32 + 1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, 1))
        self.interpo = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.refine = ref_seg()
        self._init_weight()

    def forward(self, input1, input2):
        input1_1, input2_1, input3_1, input4_1, input5_1 = input1[0], input1[1], input1[2], input1[3], input1[4]
        input1_2, input2_2, input3_2, input4_2, input5_2 = input2[0], input2[1], input2[2], input2[3], input2[4]

        x = torch.cat((input5_1, input5_2), dim=1)
        x = self.bam(x)
        x = self.db1(x)

        # 512*16*16
        al1, x = self.db2(input4_1, x, input4_2)  # 256*32*32
        al2, x = self.db3(input3_1, x, input3_2)  # 128*64*64
        al3, x = self.db4(input2_1, x, input2_2)  # 64*128*128
        al4, x = self.db5(input1_1, x, input1_2)  # 32*256*256

        edge = self.classifier1(x)
        seg = self.classifier2(torch.cat((x, self.interpo(al4)), 1))
        result = self.refine(x, seg, edge)
        return al1, al2, al3, al4, result, seg

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


