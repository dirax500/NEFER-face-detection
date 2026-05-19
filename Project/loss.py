from __future__ import annotations
"""
loss.py — Detection Loss for Event-Based Face Detection

Combines:
  • Focal loss on objectness (handles severe pos/neg imbalance)
  • CIoU regression loss on matched positive anchors
  • Two-scale support (P4 and P5 from FPN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import _box_iou


# ── IoU variants ──────────────────────────────────────────────────────────────

def box_ciou(pred_xyxy: torch.Tensor,
             gt_xyxy:   torch.Tensor) -> torch.Tensor:
    """
    Complete IoU loss for a set of matched box pairs.

    Args:
        pred_xyxy : [N, 4]  predicted boxes in xyxy normalised coords
        gt_xyxy   : [N, 4]  ground-truth boxes in xyxy normalised coords

    Returns:
        ciou_loss : scalar
    """
    pw = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=1e-6)
    ph = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=1e-6)
    gw = (gt_xyxy[:, 2]   - gt_xyxy[:, 0]).clamp(min=1e-6)
    gh = (gt_xyxy[:, 3]   - gt_xyxy[:, 1]).clamp(min=1e-6)

    pcx = pred_xyxy[:, 0] + pw / 2
    pcy = pred_xyxy[:, 1] + ph / 2
    gcx = gt_xyxy[:,  0] + gw / 2
    gcy = gt_xyxy[:,  1] + gh / 2

    # Intersection
    ix1 = torch.max(pred_xyxy[:, 0], gt_xyxy[:, 0])
    iy1 = torch.max(pred_xyxy[:, 1], gt_xyxy[:, 1])
    ix2 = torch.min(pred_xyxy[:, 2], gt_xyxy[:, 2])
    iy2 = torch.min(pred_xyxy[:, 3], gt_xyxy[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    union = pw * ph + gw * gh - inter + 1e-6
    iou   = inter / union

    # Diagonal of enclosing box
    ex1 = torch.min(pred_xyxy[:, 0], gt_xyxy[:, 0])
    ey1 = torch.min(pred_xyxy[:, 1], gt_xyxy[:, 1])
    ex2 = torch.max(pred_xyxy[:, 2], gt_xyxy[:, 2])
    ey2 = torch.max(pred_xyxy[:, 3], gt_xyxy[:, 3])
    c2  = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-6

    # Centre distance
    d2  = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

    # Aspect-ratio consistency term
    v    = (4 / (torch.pi ** 2)) * (
               torch.atan(gw / gh) - torch.atan(pw / ph)
           ) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-6)

    ciou = iou - d2 / c2 - alpha * v
    return (1 - ciou).mean()


# ── Focal Loss ────────────────────────────────────────────────────────────────

def focal_loss(pred_logits: torch.Tensor,
               targets:     torch.Tensor,
               alpha: float = 0.25,
               gamma: float = 2.0) -> torch.Tensor:
    """Binary focal loss. Both tensors shape [N]."""
    p   = torch.sigmoid(pred_logits)
    ce  = F.binary_cross_entropy_with_logits(pred_logits, targets,
                                              reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * focal_weight * ce).mean()


# ── Anchor Matcher ────────────────────────────────────────────────────────────

def match_anchors(gt_boxes_xyxy: torch.Tensor,
                  anchors_cxcywh: torch.Tensor,
                  pos_iou_thresh: float = 0.4,
                  neg_iou_thresh: float = 0.2
                  ) :
    """
    Match one ground-truth box to anchors via IoU threshold.

    Args:
        gt_boxes_xyxy   : [4]        single GT box  (x1 y1 x2 y2)
        anchors_cxcywh  : [A, 4]     anchor boxes   (cx cy w h)
        pos/neg thresholds

    Returns:
        pos_mask   : [A]  bool
        neg_mask   : [A]  bool
        ignore_mask: [A]  bool  (between thresholds)
    """
    # Convert anchors to xyxy
    ax1 = anchors_cxcywh[:, 0] - anchors_cxcywh[:, 2] / 2
    ay1 = anchors_cxcywh[:, 1] - anchors_cxcywh[:, 3] / 2
    ax2 = anchors_cxcywh[:, 0] + anchors_cxcywh[:, 2] / 2
    ay2 = anchors_cxcywh[:, 1] + anchors_cxcywh[:, 3] / 2
    anch_xyxy = torch.stack([ax1, ay1, ax2, ay2], dim=1)

    iou = _box_iou(gt_boxes_xyxy.unsqueeze(0), anch_xyxy)[0]  # [A]

    pos_mask    = iou >= pos_iou_thresh
    neg_mask    = iou <  neg_iou_thresh
    ignore_mask = ~pos_mask & ~neg_mask

    # Ensure at least one positive (best-matching anchor)
    if pos_mask.sum() == 0:
        best = iou.argmax()
        pos_mask[best] = True
        neg_mask[best] = False
        ignore_mask[best] = False

    return pos_mask, neg_mask, ignore_mask


def encode_box(gt_xyxy: torch.Tensor,
               anchor_cxcywh: torch.Tensor) -> torch.Tensor:
    """
    Encode gt box relative to anchor for regression target.
    Returns [tx, ty, tw, th] such that sigmoid(tx)*aw*2 + ... ≈ gt.
    """
    gcx = (gt_xyxy[0] + gt_xyxy[2]) / 2
    gcy = (gt_xyxy[1] + gt_xyxy[3]) / 2
    gw  = gt_xyxy[2] - gt_xyxy[0]
    gh  = gt_xyxy[3] - gt_xyxy[1]

    acx, acy, aw, ah = (anchor_cxcywh[i] for i in range(4))

    # Inverse of decode: sigmoid(tx)*aw*2 + acx - aw/2 = gcx
    tx = torch.logit(((gcx - acx + aw / 2) / (aw * 2)).clamp(1e-4, 1 - 1e-4))
    ty = torch.logit(((gcy - acy + ah / 2) / (ah * 2)).clamp(1e-4, 1 - 1e-4))
    # w/h encoding: sigmoid(tw) = sqrt(gw / (aw * 2)) clamped
    tw = torch.logit((torch.sqrt(gw / (aw * 2 + 1e-6))).clamp(1e-4, 1 - 1e-4))
    th = torch.logit((torch.sqrt(gh / (ah * 2 + 1e-6))).clamp(1e-4, 1 - 1e-4))

    return torch.stack([tx, ty, tw, th])


# ── Main Loss ──────────────────────────────────────────────────────────────────

class DetectionLoss(nn.Module):
    """
    Two-scale detection loss.

    Args:
        lambda_box  : weight on regression term (default 5.0)
        pos_iou     : IoU threshold to consider an anchor positive
        neg_iou     : IoU threshold below which an anchor is negative
    """

    def __init__(self,
                 lambda_box: float = 5.0,
                 pos_iou:    float = 0.4,
                 neg_iou:    float = 0.2):
        super().__init__()
        self.lambda_box = lambda_box
        self.pos_iou    = pos_iou
        self.neg_iou    = neg_iou

    def _loss_for_scale(self,
                        preds:     torch.Tensor,
                        anchors:   torch.Tensor,
                        gt_batch:  torch.Tensor,
                        img_h: int, img_w: int
                        ) :
        """
        Args:
            preds    : [B, A, fh, fw, 5]
            anchors  : [fh*fw*A, 4]  cx cy w h  normalised
            gt_batch : [B, 4]        x1 y1 x2 y2 normalised
        """
        B, n_a, fh, fw, _ = preds.shape
        A_total = n_a * fh * fw
        device  = preds.device

        # Flatten predictions: [B, A_total, 5]
        p = preds.view(B, n_a, fh * fw, 5)
        p = p.permute(0, 2, 1, 3).contiguous().view(B, A_total, 5)

        conf_loss_total = torch.tensor(0.0, device=device)
        box_loss_total  = torch.tensor(0.0, device=device)
        n_pos_total     = 0

        for b in range(B):
            gt    = gt_batch[b]              # [4]
            p_b   = p[b]                     # [A_total, 5]
            confs = p_b[:, 0]
            boxes = p_b[:, 1:]

            pos_mask, neg_mask, _ = match_anchors(
                gt, anchors, self.pos_iou, self.neg_iou
            )

            # Confidence targets: 1 for positives, 0 for negatives
            conf_target = torch.zeros(A_total, device=device)
            conf_target[pos_mask] = 1.0

            # Use only pos+neg (ignore ambiguous)
            used = pos_mask | neg_mask
            conf_loss_total = conf_loss_total + focal_loss(
                confs[used], conf_target[used]
            )

            # Regression only on positives
            n_pos = pos_mask.sum().item()
            if n_pos > 0:
                pos_anch = anchors[pos_mask]           # [P, 4]
                pos_preds = boxes[pos_mask]            # [P, 4] raw offsets

                # Decode predicted boxes → xyxy
                acx, acy, aw, ah = (pos_anch[:, i] for i in range(4))
                tx, ty, tw, th = (pos_preds[:, i] for i in range(4))

                pcx = torch.sigmoid(tx) * aw * 2 + acx - aw * 0.5
                pcy = torch.sigmoid(ty) * ah * 2 + acy - ah * 0.5
                pw  = aw  * (2 * torch.sigmoid(tw)) ** 2
                ph  = ah  * (2 * torch.sigmoid(th)) ** 2

                px1 = (pcx - pw / 2).clamp(0, 1)
                py1 = (pcy - ph / 2).clamp(0, 1)
                px2 = (pcx + pw / 2).clamp(0, 1)
                py2 = (pcy + ph / 2).clamp(0, 1)
                pred_xyxy = torch.stack([px1, py1, px2, py2], dim=1)

                gt_rep = gt.unsqueeze(0).expand(n_pos, -1)
                box_loss_total = box_loss_total + box_ciou(pred_xyxy, gt_rep)
                n_pos_total   += n_pos

        n = max(B, 1)
        return conf_loss_total / n, box_loss_total / n

    def forward(self,
                preds_s1:  torch.Tensor,
                preds_s2:  torch.Tensor,
                anchors_s1: torch.Tensor,
                anchors_s2: torch.Tensor,
                gt_batch:  torch.Tensor,
                img_h: int = 260,
                img_w: int = 346
                ) :
        """
        Args:
            preds_s1/s2  : detector head outputs [B, A, fh, fw, 5]
            anchors_s1/s2: anchor tensors from model.get_anchors()
            gt_batch     : [B, 4]  x1 y1 x2 y2 normalised
        """
        conf1, box1 = self._loss_for_scale(preds_s1, anchors_s1, gt_batch,
                                           img_h, img_w)
        conf2, box2 = self._loss_for_scale(preds_s2, anchors_s2, gt_batch,
                                           img_h, img_w)

        conf_loss = conf1 + conf2
        box_loss  = box1  + box2
        total     = conf_loss + self.lambda_box * box_loss

        return {
            'total'    : total,
            'conf_loss': conf_loss.detach(),
            'box_loss' : box_loss.detach(),
        }
