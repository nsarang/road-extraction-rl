import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from skimage.morphology import binary_dilation, disk


def boundary_evaluation(prediction, target, num_classes):
    predictions_frames = prediction.cpu().numpy()
    groundtruth_frames = target.cpu().numpy()
    results = np.stack(
        [
            precision_recall_fscore_boundary(
                pred_frame == class_idx, gt_frame == class_idx
            )
            for class_idx in range(num_classes)
            for pred_frame, gt_frame in zip(
                predictions_frames, groundtruth_frames
            )
        ],
        axis=1,
    )
    F, precision, recall = results.mean(axis=-1)
    meters = {
        "F": F,
        "precision": precision,
        "recall": recall,
    }
    return meters


def precision_recall_fscore_boundary(
    pred_mask, gt_mask, ignore_index=None, bound_th=0.008
):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        pred_mask       (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.

    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """

    bound_pix = (
        bound_th
        if bound_th >= 1
        else np.ceil(bound_th * np.linalg.norm(pred_mask.shape))
    )

    if ignore_index is not None:
        pred_mask[ignore_index] = 0
        gt_mask[ignore_index] = 0

    # Get the pixel boundaries of both masks
    fg_boundary = mask2boundary(pred_mask)
    gt_boundary = mask2boundary(gt_mask)

    fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary, disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    #% Compute precision and recall
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
    return F, precision, recall


def mask2boundary(mask):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.

     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    mask = np.clip(mask, 0, 1).astype(bool)

    e = np.zeros_like(mask)
    s = np.zeros_like(mask)
    se = np.zeros_like(mask)

    e[:, :-1] = mask[:, 1:]
    s[:-1, :] = mask[1:, :]
    se[:-1, :-1] = mask[1:, 1:]

    bmap = mask ^ e | mask ^ s | mask ^ se
    bmap[-1, :] = mask[-1, :] ^ e[-1, :]
    bmap[:, -1] = mask[:, -1] ^ s[:, -1]
    bmap[-1, -1] = 0

    return bmap
