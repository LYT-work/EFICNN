import torch
import torch.nn as nn
import torch.nn.functional as F


def BCEDiceLoss(inputs, targets):
    if targets.dim() == 4:
        targets = torch.squeeze(targets, dim=1)
    targets = F.one_hot(targets.long(), num_classes=2)
    targets = targets.permute(0, 3, 1, 2).float()
    criterion = nn.BCEWithLogitsLoss()
    bce = criterion(inputs, targets)

    smooth = 1e-5
    input = torch.sigmoid(inputs)
    num = targets.shape[0]
    input = input.reshape(num, -1)
    target = targets.reshape(num, -1)
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    labuda = 1.00
    loss = labuda * bce + (1-labuda) * dice
    return loss


def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


def weight_binary_cross_entropy_loss(input,target):
    n, c, _, _ = input.shape
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    target = F.one_hot(target.long(), num_classes=2)
    target =target.permute(0, 3, 1, 2).float()

    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(input,target)

    return loss


# Boundary loss for RS images
def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    divce = label.device
    one_hot_label = torch.eye(
        n_classes, device='cuda', requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label


def BoundaryLoss(pred,gt,theta0=3, theta=3):
    "return boundary loss"
    n, c, _, _ = pred.shape

    # softmax so that predicted map can be distributed in [0, 1]
    pred = torch.softmax(pred, dim=1)
    if gt.dim() == 4:
        target = torch.squeeze(gt, dim=1)
    gt = F.one_hot(gt.long(), num_classes=2)
    gt =gt.permute(0, 3, 1, 2).float()

    # boundary map
    gt_b = F.max_pool2d(
        1 - gt, kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
    gt_b -= 1 - gt

    pred_b = F.max_pool2d(
        1 - pred, kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
    pred_b -= 1 - pred

    # extended boundary map
    gt_b_ext = F.max_pool2d(
        gt_b, kernel_size=theta, stride=1, padding=(theta - 1) // 2)

    pred_b_ext = F.max_pool2d(
        pred_b, kernel_size=theta, stride=1, padding=(theta - 1) // 2)

    # reshape
    gt_b = torch.flatten(gt_b,1)
    pred_b =torch.flatten(pred_b,1)
    gt_b_ext = torch.flatten(gt_b_ext,1)
    pred_b_ext = torch.flatten(pred_b_ext,1)

    # Precision, Recall
    P = torch.sum(pred_b * gt_b_ext) / (torch.sum(pred_b) + 1e-7)
    R = torch.sum(pred_b_ext * gt_b) / (torch.sum(gt_b) + 1e-7)

    # Boundary F1 Score
    BF1 = 2 * P * R / (P + R + 1e-7)
    loss = torch.mean(1 - BF1)

    return loss


class Boundary_ce_loss(nn.Module):
    def __init__(self, theta0=3, theta=3, weight=None, reduction='mean',ignore_index=255):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta
        self.weight=weight
        self.reduction=reduction
        self.ignore_index=ignore_index

    def forward(self, input, target,epoch_id,epoch_max):
        w0 = 0.05
        if epoch_id < int(epoch_max/2):
            w =0
            ce_loss = cross_entropy(input = input,target=target)
            loss = (1 - w) * ce_loss

        else:
            w = w0+0.01*(epoch_id-int(epoch_max/2))
            ce_loss = cross_entropy(input=input, target=target)
            boundary_loss = BoundaryLoss(pred=input, gt=target, theta0=3, theta=3)
            if w > 0.5:
                w = 0.5
            loss = (1 - w) * ce_loss + w * boundary_loss

        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1e-5
        input = torch.sigmoid(input)
        n, c, _, _ = input.shape
        mask = target[0, 0, :, :]
        reversed_mask = mask
        reversed_mask = torch.where(reversed_mask == 1, 0, 1)
        i_m_reverse = reversed_mask
        mask_join = torch.zeros(size=(n, 2, mask.shape[0], mask.shape[1]))
        mask_join[:,0,...] = mask
        mask_join[:,1,...] = i_m_reverse
        mask_join = mask_join.to(device='cuda')

        num = mask_join.shape[0]
        input = input.reshape(num, -1)
        target = mask_join.reshape(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        label = label.float()
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)

        ce_loss = self.crit(logits, label)

        focal_loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        if self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        return focal_loss


class FocalLoss_with_dice(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss_with_dice, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        n, c, _, _ = logits.shape

        if label.dim() == 4:
            label = torch.squeeze(label, dim=1)
        target = F.one_hot(label.long(), num_classes=2)
        target = target.permute(0, 3, 1, 2).float()

        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[target == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(target == 1, probs, 1 - probs)

        ce_loss = self.crit(logits, target)

        focal_loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        if self.reduction == 'sum':
            focal_loss = focal_loss.sum()

        smooth=1e-5
        num=label.shape[0]
        probs=probs.reshape(num,-1)
        target=target.reshape(num,-1)
        intersection=(probs*(target.float()))
        dice=(2.*intersection.sum(1)+smooth)/(probs.sum(1)+(target.float()).sum(1)+smooth)
        dice=1-dice.sum()/num
        loss = 0.5*focal_loss+0.5*dice
        return loss


class Focal_Dice_BL(nn.Module):
    def __init__(self):
        super(Focal_Dice_BL, self).__init__()

        self.Focal_loss = FocalLoss()
        self.Dice_loss = DiceLoss()

    def forward(self,input, target,epoch_id,epoch_max):
        if epoch_id < int(epoch_max/2):
            focal = self.Focal_loss(logits=input, label=target)
            dice = self.Dice_loss(input=input, target=target)
            loss = focal + 0.75*dice
        else:
            focal = self.Focal_loss(logits=input, label=target)
            dice = self.Dice_loss(input=input, target=target)
            boundary = BoundaryLoss(pred=input, gt=target, theta0=3, theta=3)
            w0=0.05
            w = w0+0.01*(epoch_id-int(epoch_max/2))
            if w>0.5:
                w = 0.5
            loss = (1-w)*(focal + 0.75*dice)+ w*boundary

        return loss