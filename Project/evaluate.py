from __future__ import annotations
"""
evaluate.py — Detection Metrics

Computes:
  • IoU between predicted and ground-truth boxes
  • Precision / Recall at IoU threshold
  • mAP@0.5
  • Average inference latency
"""

import time
import torch
import numpy as np
from collections import defaultdict


def box_iou_single(pred_xyxy: torch.Tensor,
                   gt_xyxy:   torch.Tensor) -> float:
    """IoU between two boxes. Both [4] xyxy normalised."""
    ix1 = max(pred_xyxy[0].item(), gt_xyxy[0].item())
    iy1 = max(pred_xyxy[1].item(), gt_xyxy[1].item())
    ix2 = min(pred_xyxy[2].item(), gt_xyxy[2].item())
    iy2 = min(pred_xyxy[3].item(), gt_xyxy[3].item())
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    pw = pred_xyxy[2].item() - pred_xyxy[0].item()
    ph = pred_xyxy[3].item() - pred_xyxy[1].item()
    gw = gt_xyxy[2].item()   - gt_xyxy[0].item()
    gh = gt_xyxy[3].item()   - gt_xyxy[1].item()
    union = pw * ph + gw * gh - inter + 1e-6
    return inter / union


@torch.no_grad()
def evaluate(model,
             val_loader,
             device: str = 'cuda',
             conf_thresh: float = 0.3,
             iou_thresh:  float = 0.5,
             max_batches: int | None = None
             ) :
    """
    Evaluate the model on the validation DataLoader.

    Returns dict with keys: precision, recall, f1, ap50, mean_iou,
                            mean_latency_ms, mean_best_iou
    """
    model.eval()
    model.to(device)

    tp_list, fp_list, fn_list = [], [], []
    iou_list    = []
    best_ious   = []
    latencies   = []

    all_scores  = []
    all_matched = []
    all_gt_cnt  = 0

    for batch_idx, (voxel, gt_boxes) in enumerate(val_loader):
        if max_batches and batch_idx >= max_batches:
            break

        voxel     = voxel.to(device)
        gt_boxes  = gt_boxes.to(device)          # [B, 4]

        t0 = time.perf_counter()
        preds_s1, preds_s2 = model(voxel)
        torch.cuda.synchronize() if device == 'cuda' else None
        latencies.append((time.perf_counter() - t0) * 1000 / voxel.shape[0])

        fh1, fw1 = preds_s1.shape[2], preds_s1.shape[3]
        fh2, fw2 = preds_s2.shape[2], preds_s2.shape[3]
        a1, a2   = model.get_anchors(fh1, fw1, fh2, fw2, device=device)

        decoded = model.decode_predictions(preds_s1, preds_s2,
                                           conf_thresh=conf_thresh,
                                           iou_thresh=0.4)

        for b, result in enumerate(decoded):
            gt  = gt_boxes[b]   # [4] xyxy normalised
            all_gt_cnt += 1

            boxes  = result['boxes']    # [N, 4]
            scores = result['scores']   # [N]

            if boxes.shape[0] == 0:
                fn_list.append(1)
                tp_list.append(0)
                fp_list.append(0)
                best_ious.append(0.0)
                all_scores.append(torch.tensor([]))
                all_matched.append(torch.tensor([]))
                continue

            # Compute IoU with GT for each predicted box
            ious = torch.tensor([
                box_iou_single(boxes[i], gt) for i in range(boxes.shape[0])
            ])
            best_iou = ious.max().item()
            best_ious.append(best_iou)

            # Binary: at least one prediction matches GT at iou_thresh
            matched = (ious >= iou_thresh).float()
            if matched.max() > 0:
                tp_list.append(1)
                fn_list.append(0)
            else:
                tp_list.append(0)
                fn_list.append(1)

            fp_list.append(max(0, boxes.shape[0] - int(matched.max().item())))
            iou_list.extend(ious.tolist())

            all_scores.append(scores.cpu())
            all_matched.append(matched.cpu())

    # Basic metrics
    TP = sum(tp_list)
    FP = sum(fp_list)
    FN = sum(fn_list)

    precision  = TP / (TP + FP + 1e-6)
    recall     = TP / (TP + FN + 1e-6)
    f1         = 2 * precision * recall / (precision + recall + 1e-6)
    mean_iou   = float(np.mean(iou_list)) if iou_list else 0.0
    mean_best  = float(np.mean(best_ious))
    mean_lat   = float(np.mean(latencies)) if latencies else 0.0

    # AP@50 via precision-recall curve
    ap50 = _compute_ap(all_scores, all_matched, all_gt_cnt)

    return {
        'precision'      : precision,
        'recall'         : recall,
        'f1'             : f1,
        'ap50'           : ap50,
        'mean_iou'       : mean_iou,
        'mean_best_iou'  : mean_best,
        'mean_latency_ms': mean_lat,
    }


def _compute_ap(all_scores, all_matched, n_gt):
    """Simple AP computation via PR curve."""
    if n_gt == 0:
        return 0.0

    scores  = torch.cat([s for s in all_scores if len(s) > 0]) \
              if any(len(s) > 0 for s in all_scores) else torch.tensor([])
    matched = torch.cat([m for m in all_matched if len(m) > 0]) \
              if any(len(m) > 0 for m in all_matched) else torch.tensor([])

    if len(scores) == 0:
        return 0.0

    order   = scores.argsort(descending=True)
    matched = matched[order]

    tp_cum  = torch.cumsum(matched, dim=0)
    fp_cum  = torch.cumsum(1 - matched, dim=0)
    n       = torch.arange(1, len(matched) + 1, dtype=torch.float)

    prec = tp_cum / n
    rec  = tp_cum / n_gt

    # Interpolated AP (11-point)
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        mask  = rec >= t
        ap   += prec[mask].max().item() if mask.any() else 0.0
    return ap / 11.0


def print_metrics(metrics: dict, epoch: int | None = None):
    prefix = f"[Epoch {epoch}] " if epoch is not None else ""
    print(
        f"{prefix}"
        f"AP@50={metrics['ap50']:.3f}  "
        f"P={metrics['precision']:.3f}  "
        f"R={metrics['recall']:.3f}  "
        f"F1={metrics['f1']:.3f}  "
        f"mIoU={metrics['mean_best_iou']:.3f}  "
        f"lat={metrics['mean_latency_ms']:.1f}ms"
    )
