import argparse
import pprint
import cv2
import yaml
from PIL import Image
import torch.backends.cudnn as cudnn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import *
from pytorch_grad_cam.utils.image import show_cam_on_image

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

from torch.utils.data import DataLoader
from dataset.colorindex import convert_to_index_image
from util.classes import CLASSES
from util.utils import *
from dataset.mode import Dataset
from dataset.creatflow import *

parser = argparse.ArgumentParser(description='Supervised Change detection')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--model-path', type=str, required=True)
parser.add_argument('--output-dir', type=str, required=True)
device = torch.device('cuda')


def test_model(config_path, model_path, output_dir):
    cfg = yaml.load(open(config_path, "r"), Loader=yaml.Loader)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    logger.info('{}\n'.format(pprint.pformat({**cfg, 'ngpus': 1})))

    cudnn.enabled = True
    cudnn.benchmark = True

    model = EFICNN()   # model name
    model.load_state_dict(torch.load(model_path)['model'])
    model = model.cuda()
    model.eval()

    testset = Dataset(cfg['dataset'], cfg['data_root'], 'test')
    testloader = DataLoader(testset, batch_size=1, pin_memory=True, num_workers=0, drop_last=False)
    num_batches = len(testloader)
    print("Number of batches:", num_batches)

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    correct_pixel = AverageMeter()
    total_pixel = AverageMeter()

    pred_list = []
    target_list = []

    numbegan = 1
    os.makedirs(output_dir, exist_ok=True)
    for i, (imgA, imgB, mask, id) in enumerate(testloader):
        rgbA = imgA.permute(0, 2, 3, 1).squeeze(0)
        rgbB = imgB.permute(0, 2, 3, 1).squeeze(0)
        rgbA = rgbA.numpy()
        rgbB = rgbB.numpy()
        rgbA = np.float32(rgbA) / 255
        rgbB = np.float32(rgbB) / 255

        imgA = imgA.cuda()
        imgB = imgB.cuda()
        mask = mask.cuda()

        # predict and output
        out = model(imgA, imgB)
        _, predicted = torch.max(out[0].data, 1)

        predicted = predicted.cpu().numpy()
        predicted_dir = os.path.join(output_dir, 'predicted')
        os.makedirs(predicted_dir, exist_ok=True)
        output_path = os.path.join(predicted_dir, f"pred{i}.png")
        predicted_image = Image.fromarray(predicted.squeeze().astype(np.uint8))
        predicted_image = convert_to_index_image(predicted_image)
        predicted_image.save(output_path)

        predicted_dir = os.path.join(output_dir, 'color_predicted')
        os.makedirs(predicted_dir, exist_ok=True)
        output_path = os.path.join(predicted_dir, f"cpred{i}.png")
        cpred = create_cpred(predicted, mask.cpu().numpy())
        cpred_image = Image.fromarray(cpred.squeeze().astype(np.uint8))
        cpred_image = convert_to_index_image(cpred_image)
        cpred_image.save(output_path)

        pred_edge = seg2bmap(predicted)
        edge_dir = os.path.join(output_dir, 'pred_edge')
        os.makedirs(edge_dir, exist_ok=True)
        output_path = os.path.join(edge_dir, f"pred_edge{i}.png")
        pred_edge_image = Image.fromarray(pred_edge.squeeze().astype(np.uint8))
        pred_edge_image = convert_to_index_image(pred_edge_image)
        pred_edge_image.save(output_path)

        # true_edge = seg2bmap(mask.cpu().numpy())
        # edge_dir = os.path.join(output_dir, 'true_edge')
        # os.makedirs(edge_dir, exist_ok=True)
        # output_path = os.path.join(edge_dir, f"true_edge{i}.png")
        # true_edge_image = Image.fromarray(true_edge.squeeze().astype(np.uint8))
        # true_edge_image = convert_to_index_image(true_edge_image)
        # true_edge_image.save(output_path)

        # flow = os.path.join(output_dir, 'flow')
        # os.makedirs(flow, exist_ok=True)
        # flow1_path = os.path.join(flow, f"flow1_{i}.png")
        # flow2_path = os.path.join(flow, f"flow2_{i}.png")
        # flow3_path = os.path.join(flow, f"flow3_{i}.png")
        # flow3_char_path= os.path.join(flow, f"flow3_char{i}.png")
        # flow1 = flow_to_image_torch(out[1].cpu())
        # flow2 = flow_to_image_torch(out[2].cpu())
        # flow3 = flow_to_image_torch(out[3].cpu())
        # flow3_char = draw_flow(out[3].cpu())
        # flow1.save(flow1_path)
        # flow2.save(flow2_path)
        # flow3.save(flow3_path)
        # flow3_char.save(flow3_char_path)

        target_layer = [model.convout]
        cam = GradCAM(model=model, target_layers=target_layer, use_cuda=False)
        grayscale_cam = cam(input_tensorA=imgA, input_tensorB=imgB)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgbA, grayscale_cam)
        heatmap_dir = os.path.join(output_dir, 'heatmap')
        os.makedirs(heatmap_dir, exist_ok=True)
        output_filename = os.path.join(heatmap_dir, f'heatmap_{i}.png')
        cv2.imwrite(output_filename, visualization)

        # evaluate
        intersection, union, target = intersectionAndUnion(predicted, mask.cpu().numpy(), cfg['nclass'])
        intersection_meter.update(intersection)
        union_meter.update(union)
        correct_pixel.update((predicted == mask.cpu().numpy()).sum().item())
        total_pixel.update(predicted.size)

        pred_list.append(predicted)
        target_list.append(mask.cpu().numpy())

        numbegan = numbegan + 1

    pred_all = np.concatenate(pred_list)
    target_all = np.concatenate(target_list)

    kappa = compute_kappa(pred_all, target_all, cfg['nclass'])
    f1 = compute_f1(pred_all, target_all, cfg['nclass'])
    recall = compute_recall(pred_all, target_all, cfg['nclass'])
    precision = compute_precision(pred_all, target_all, cfg['nclass'])
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIoU = np.mean(iou_class)
    overall_acc = correct_pixel.sum / total_pixel.sum * 100.0

    for (cls_idx, iou) in enumerate(iou_class):
        logger.info('***** Evaluation ***** >>>> Class [{:} {:}] ''IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx],iou))
    logger.info('***** Evaluation ***** >>>> mIoU: {:.2f}'.format(mIoU))
    logger.info('***** Evaluation ***** >>>> OA: {:.2f}'.format(overall_acc))
    logger.info('***** Evaluation ***** >>>> Kappa: {:.2f}'.format(kappa))
    logger.info('***** Evaluation ***** >>>> F1: {:.2f}'.format(f1))
    logger.info('***** Evaluation ***** >>>> Precision: {:.2f}'.format(precision))
    logger.info('***** Evaluation ***** >>>> Recall: {:.2f}'.format(recall))
    logger.info('Testing completed!')


if __name__ == '__main__':
    args = parser.parse_args()
    test_model(args.config, args.model_path, args.output_dir)
