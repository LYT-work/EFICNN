import numpy as np
import logging
import os
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class AverageMeter(object):
    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K):
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def compute_kappa(output, target, num_classes):
    output = output.flatten()
    target = target.flatten()
    kappa = cohen_kappa_score(output, target)
    return kappa


def compute_f1(output, target, num_classes):
    output = output.flatten()
    target = target.flatten()

    if num_classes > 2:
        f1 = f1_score(output, target, average='macro', labels=np.arange(num_classes))
    else:
        f1 = f1_score(output, target)

    f1 = f1 * 100
    f1 = round(f1, 2)
    return f1


def compute_precision(output, target, num_classes):
    output = output.flatten()
    target = target.flatten()

    if num_classes > 2:
        precision = precision_score(output, target, average='macro', labels=np.arange(num_classes))
    else:
        precision = precision_score(output, target)

    precision = precision * 100
    precision = round(precision, 2)
    return precision


def compute_recall(output, target, num_classes):
    output = output.flatten()
    target = target.flatten()

    if num_classes > 2:
        recall = recall_score(output, target, average='macro', labels=np.arange(num_classes))
    else:
        recall = recall_score(output, target)

    recall = recall * 100
    recall = round(recall, 2)
    return recall


def create_cpred(predicted, target):
    batch_size, height, width = predicted.shape
    fpfn_mask = np.zeros((batch_size, height, width), dtype=np.uint8)

    fp_mask = np.logical_and(predicted == 1, target == 0)
    fpfn_mask[fp_mask] = 2

    fn_mask = np.logical_and(predicted == 0, target == 1)
    fpfn_mask[fn_mask] = 3

    result = np.copy(predicted)
    result[fpfn_mask == 2] = 0
    result[fpfn_mask == 3] = 0
    result = fpfn_mask + result

    return result


def seg2bmap(seg):
    seg = np.transpose(seg, (1, 2, 0))
    assert np.atleast_3d(seg).shape[2] == 1

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    return b


def db_eval_boundary(pred_mask, gt_mask):
    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(pred_mask)
    gt_boundary = seg2bmap(gt_mask)

    # Get the intersection
    gt_match = gt_boundary * fg_boundary
    fg_match = fg_boundary * gt_boundary

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)
    return F



logs = set()
def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

