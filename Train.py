import argparse
import pprint
import yaml
import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from thop import profile
from torchstat import stat

from model.EFICNN import EFICNN
# from compares.models.A2Net import A2Net
# from compares.models.AERNet import AERNet
# from compares.models.BIT import BASE_Transformer
# from compares.models.ChangeFomer import ChangeFormerV6
# from compares.models.DESSN import DESSN
# from compares.models.DSIFN import DSIFN
# from compares.models.SiamUnetEF import SiamUnetEF
# from compares.models.SiamUnetdiff import SiamUnetdiff
# from compares.models.SiamUnetconc import SiamUnetconc
# from compares.models.SNUNet import SNUNet
# from compares.models.TCDNet import TCDNet
# from compares.models.USSFCNet import USSFCNet

from dataset.mode import Dataset
from util.classes import CLASSES
from util.loss import BCEDiceLoss
from util.utils import *
device = torch.device('cuda')

parser = argparse.ArgumentParser(description='Supervised Change Detection')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)


def evaluate(model, loader, cfg):
    model.eval()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    correct_pixel = AverageMeter()
    total_pixel = AverageMeter()

    pred_list = []
    target_list = []

    with torch.no_grad():
        for imgA, imgB, mask, id in loader:
            imgA = imgA.to('cuda')
            imgB = imgB.to('cuda')

            out = model(imgA, imgB)
            pred = out[0].argmax(dim=1)

            intersection, union, target = intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'])
            intersection_meter.update(intersection)
            union_meter.update(union)
            correct_pixel.update((pred.cpu() == mask).sum().item())
            total_pixel.update(pred.numel())

            pred_list.append(pred.cpu().numpy())
            target_list.append(mask.numpy())

    pred_all = np.concatenate(pred_list)
    target_all = np.concatenate(target_list)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)
    overall_acc = correct_pixel.sum / total_pixel.sum * 100.0
    kappa = compute_kappa(pred_all, target_all, cfg['nclass'])
    f1 = compute_f1(pred_all, target_all, cfg['nclass'])
    precision = compute_precision(pred_all, target_all, cfg['nclass'])
    recall = compute_recall(pred_all, target_all, cfg['nclass'])
    return mIOU, iou_class, overall_acc, kappa, f1, precision, recall


def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    all_args = {**cfg, **vars(args), 'ngpus': 1}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True
    model = EFICNN()  # model name
    model = model.to(device)

    # stat(model, (3, 256, 256))
    # input_data1 = torch.randn(1, 3, 256, 256).to(device)
    # input_data2 = torch.randn(1, 3, 256, 256).to(device)

    # flops, params = profile(model, inputs=(input_data1, input_data2))
    # logger.info('Parameters: {:.2f}M'.format(params/1e6))
    # logger.info('FLOPs: {:.2f}G'.format(flops/1e9))

    optimizer = SGD(model.parameters(), cfg['lr'], momentum=0.9, weight_decay=1e-4)
    trainset = Dataset(cfg['dataset'], cfg['data_root'], 'train')
    valset = Dataset(cfg['dataset'], cfg['data_root'], 'val')
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=0, drop_last=False)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=0, drop_last=False)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    (previous_best_iou, previous_best_acc, previous_best_kappa, previous_best_f1, previous_best_precision,
     previous_best_recall) = 0.00, 0.00, 0.00, 0.00, 0.00, 0.00
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best_iou = checkpoint['previous_best_iou']
        previous_best_acc = checkpoint['previous_best_acc']
        previous_best_kappa = checkpoint['previous_best_kappa']
        previous_best_f1 = checkpoint['previous_best_f1']
        previous_best_precision = checkpoint['previous_best_precision']
        previous_best_recall = checkpoint['previous_best_recall']
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous Best mIoU: {:.2f}, OA: {:.2f}, '
                    'Kappa: {:.2f}, F1: {:.2f}, Precision: {:.2f}, Recall: {:.2f}'.format(epoch, optimizer.param_groups[0]['lr'],
                   previous_best_iou, previous_best_acc, previous_best_kappa, previous_best_f1, previous_best_precision, previous_best_recall))

        model.train()
        total_loss = AverageMeter()

        for i, (imgA, imgB, mask) in enumerate(trainloader):
            imgA, imgB, mask = imgA.cuda(), imgB.cuda(), mask.cuda()

            out = model(imgA, imgB)
            loss = BCEDiceLoss(out[0], mask).cuda()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            if (i % (max(2, len(trainloader) // 8)) == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        mIoU, iou_class, overall_acc, kappa, f1, precision, recall = evaluate(model, valloader, cfg)

        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] ''IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
        logger.info('***** Evaluation ***** >>>> mIoU: {:.2f}'.format(mIoU))
        logger.info('***** Evaluation ***** >>>> OA: {:.2f}'.format(overall_acc))
        logger.info('***** Evaluation ***** >>>> Kappa: {:.2f}'.format(kappa))
        logger.info('***** Evaluation ***** >>>> F1: {:.2f}'.format(f1))
        logger.info('***** Evaluation ***** >>>> Precision: {:.2f}'.format(precision))
        logger.info('***** Evaluation ***** >>>> Recall: {:.2f}\n'.format(recall))

        is_best = f1 > previous_best_f1
        previous_best_f1 = max(f1, previous_best_f1)
        if is_best:
            previous_best_iou = mIoU
            previous_best_acc = overall_acc
            previous_best_kappa = kappa
            previous_best_precision = precision
            previous_best_recall = recall
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best_iou': previous_best_iou,
            'previous_best_acc': previous_best_acc,
            'previous_best_kappa': previous_best_kappa,
            'previous_best_f1': previous_best_f1,
            'previous_best_precision': previous_best_precision,
            'previous_best_recall': previous_best_recall,
        }
        torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))

if __name__ == '__main__':
    main()