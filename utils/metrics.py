"""Segmentation metrics: Dice, sensitivity, Hausdorff. [PROPOSAL metrics]"""
import numpy as np
import torch
from scipy.ndimage import label as nd_label


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    """Binary Dice. pred and target are binary arrays."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = (pred & target).sum()
    return float(2 * intersection / (pred.sum() + target.sum() + eps))


def tumor_sensitivity(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Tumor-wise sensitivity: fraction of connected tumor components in target
    that have >=1 predicted voxel overlapping. [PROPOSAL] emphasis on small tumors.
    """
    labeled, n_tumors = nd_label(target.astype(bool))
    if n_tumors == 0:
        return float("nan")
    detected = 0
    for tumor_id in range(1, n_tumors + 1):
        component = labeled == tumor_id
        if (pred.astype(bool) & component).any():
            detected += 1
    return detected / n_tumors


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """95th-percentile Hausdorff distance (voxels). [PROPOSAL optional metric]"""
    from scipy.spatial.distance import directed_hausdorff
    pred_pts = np.argwhere(pred.astype(bool))
    tgt_pts = np.argwhere(target.astype(bool))
    if len(pred_pts) == 0 or len(tgt_pts) == 0:
        return float("inf")
    d1 = directed_hausdorff(pred_pts, tgt_pts)[0]
    d2 = directed_hausdorff(tgt_pts, pred_pts)[0]
    return max(d1, d2)


def stratified_sensitivity(
    pred: np.ndarray,
    target: np.ndarray,
    size_thresholds_vox: list[int] = [100, 1000],
) -> dict:
    """
    [PROPOSAL] Sensitivity stratified by tumor size.
    Returns dict with keys: 'small', 'medium', 'large'.
    size_thresholds_vox: [small_max, medium_max]; large = above medium_max.
    """
    labeled, n = nd_label(target.astype(bool))
    buckets = {"small": [], "medium": [], "large": []}
    for tid in range(1, n + 1):
        comp = labeled == tid
        sz = comp.sum()
        hit = int((pred.astype(bool) & comp).any())
        if sz <= size_thresholds_vox[0]:
            buckets["small"].append(hit)
        elif sz <= size_thresholds_vox[1]:
            buckets["medium"].append(hit)
        else:
            buckets["large"].append(hit)

    return {
        k: (float(np.mean(v)) if v else float("nan"))
        for k, v in buckets.items()
    }
