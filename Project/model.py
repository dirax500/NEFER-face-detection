from __future__ import annotations
"""
model.py — Lightweight Event-Based Face Detector
Trained from scratch (random init) on NEFER voxel-grid inputs.

Architecture overview
─────────────────────
Input  : [B, C_in, H, W]  where C_in = T*2 (default 10)
Backbone: MobileNet-style depthwise-separable conv stack
Neck   : Feature Pyramid Network (2 scales) for multi-scale detection
Head   : Per-anchor [conf, cx, cy, w, h] predictions at each scale

Anchors are hand-designed for face aspect ratios at typical distances.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building Blocks ────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, groups=1, act=True):
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride, pad,
                              groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)
        self.act  = nn.ReLU6(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    """Depthwise-Separable Conv: 8-9× fewer params than standard conv."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = ConvBnRelu(in_ch, in_ch, k=3, stride=stride, groups=in_ch)
        self.pw = ConvBnRelu(in_ch, out_ch, k=1)

    def forward(self, x):
        return self.pw(self.dw(x))


class InvertedResidual(nn.Module):
    """MobileNetV2-style block with residual connection."""
    def __init__(self, in_ch, out_ch, stride=1, expand=4):
        super().__init__()
        mid = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)
        self.block = nn.Sequential(
            ConvBnRelu(in_ch, mid, k=1),
            ConvBnRelu(mid,  mid, k=3, stride=stride, groups=mid),
            ConvBnRelu(mid, out_ch, k=1, act=False),
        )
        self.bn_out = nn.BatchNorm2d(out_ch, momentum=0.01, eps=1e-3)

    def forward(self, x):
        out = self.bn_out(self.block(x))
        if self.use_res:
            out = out + x
        return F.relu6(out, inplace=True)


# ── Anchor Definition ──────────────────────────────────────────────────────────

def make_anchors(feature_h: int, feature_w: int,
                 img_h: int, img_w: int,
                 anchor_wh: list) -> torch.Tensor:
    """
    Build anchor boxes in [cx, cy, w, h] normalised to [0,1].

    Returns: [feature_h * feature_w * n_anchors, 4]
    """
    stride_h = img_h / feature_h
    stride_w = img_w / feature_w

    grid_y = (torch.arange(feature_h).float() + 0.5) * stride_h / img_h
    grid_x = (torch.arange(feature_w).float() + 0.5) * stride_w / img_w

    cy, cx = torch.meshgrid(grid_y, grid_x, indexing='ij')  # [fh, fw]

    anchors = []
    for (aw, ah) in anchor_wh:
        aw_norm = aw / img_w
        ah_norm = ah / img_h
        w_t = torch.full_like(cx, aw_norm)
        h_t = torch.full_like(cy, ah_norm)
        anchors.append(torch.stack([cx, cy, w_t, h_t], dim=-1))

    anchors = torch.stack(anchors, dim=2)     # [fh, fw, n_a, 4]
    return anchors.view(-1, 4)                # [fh*fw*n_a, 4]


# ── Detection Head ─────────────────────────────────────────────────────────────

class DetectionHead(nn.Module):
    """
    Shared conv head for one feature-map scale.
    Predicts [conf_logit, tx, ty, tw, th] per anchor.
    """
    def __init__(self, in_ch: int, n_anchors: int, hidden: int = 128):
        super().__init__()
        self.n_anchors = n_anchors
        self.head = nn.Sequential(
            ConvBnRelu(in_ch, hidden, k=3),
            ConvBnRelu(hidden, hidden, k=3),
            nn.Conv2d(hidden, n_anchors * 5, 1),
        )

    def forward(self, x):
        out = self.head(x)                       # [B, A*5, h, w]
        B, _, fh, fw = out.shape
        out = out.view(B, self.n_anchors, 5, fh, fw)
        out = out.permute(0, 1, 3, 4, 2).contiguous()  # [B, A, fh, fw, 5]
        return out


# ── Main Model ─────────────────────────────────────────────────────────────────

class EventFaceDetector(nn.Module):
    """
    Event-based face detector.  Trains from random weights on NEFER.

    Args:
        in_channels : C_in = T * 2  (default 10 for T=5)
        img_h / img_w : sensor resolution (default 260 x 346)
    """

    # Anchors designed for human face sizes at camera distances used in NEFER.
    # Two scales: large features (/16) and small features (/32).
    ANCHORS_S1 = [(80, 100), (120, 150), (160, 200)]   # /16 scale (larger faces)
    ANCHORS_S2 = [(40,  50), ( 60,  75), ( 90, 110)]   # /32 scale (smaller faces)

    def __init__(self,
                 in_channels: int = 10,
                 img_h: int = 260,
                 img_w: int = 346):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.n_anchors = len(self.ANCHORS_S1)   # same count for both scales

        # ── Stem ─────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            ConvBnRelu(in_channels, 32, k=3, stride=2),   # /2
            DSConv(32, 64, stride=2),                      # /4
        )

        # ── Backbone stages ───────────────────────────────────────────────────
        self.stage1 = nn.Sequential(           # /8
            InvertedResidual(64,  128, stride=2, expand=4),
            InvertedResidual(128, 128, stride=1, expand=4),
        )
        self.stage2 = nn.Sequential(           # /16  → P4 (anchor scale 1)
            InvertedResidual(128, 256, stride=2, expand=4),
            InvertedResidual(256, 256, stride=1, expand=4),
            InvertedResidual(256, 256, stride=1, expand=4),
        )
        self.stage3 = nn.Sequential(           # /32  → P5 (anchor scale 2)
            InvertedResidual(256, 512, stride=2, expand=4),
            InvertedResidual(512, 512, stride=1, expand=4),
        )

        # ── FPN Neck ──────────────────────────────────────────────────────────
        self.lat_p5 = ConvBnRelu(512, 256, k=1)
        self.lat_p4 = ConvBnRelu(256, 256, k=1)
        self.fpn_p4 = ConvBnRelu(256, 256, k=3)
        self.fpn_p5 = ConvBnRelu(512, 256, k=3)

        # ── Detection Heads ───────────────────────────────────────────────────
        self.head_s1 = DetectionHead(256, self.n_anchors, hidden=128)  # /16
        self.head_s2 = DetectionHead(256, self.n_anchors, hidden=128)  # /32

        self._init_weights()

    # ── Weight init ───────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Bias conf logit toward low confidence at start → stable training
        prior_prob = 0.01
        bias_val   = -math.log((1 - prior_prob) / prior_prob)
        for head in [self.head_s1, self.head_s2]:
            last_conv = head.head[-1]
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                # conf logit → negative bias; bbox offsets → zero
                bias = torch.zeros(self.n_anchors * 5)
                for i in range(self.n_anchors):
                    bias[i * 5] = bias_val
                last_conv.bias.data.copy_(bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : [B, C_in, H, W]

        Returns:
            preds_s1 : [B, n_a, fh1, fw1, 5]   raw logits / offsets at /16
            preds_s2 : [B, n_a, fh2, fw2, 5]   raw logits / offsets at /32
        """
        c0 = self.stem(x)       # /4
        c1 = self.stage1(c0)    # /8
        c2 = self.stage2(c1)    # /16  P4
        c3 = self.stage3(c2)    # /32  P5

        # FPN top-down
        p5 = self.fpn_p5(c3)
        p4 = self.fpn_p4(
            self.lat_p4(c2) +
            F.interpolate(self.lat_p5(c3),
                          size=c2.shape[-2:], mode='nearest')
        )

        preds_s1 = self.head_s1(p4)   # /16
        preds_s2 = self.head_s2(p5)   # /32

        return preds_s1, preds_s2

    # ── Anchor helpers ────────────────────────────────────────────────────────

    def get_anchors(self, feat_h1, feat_w1,
                    feat_h2, feat_w2,
                    device='cpu') :
        a1 = make_anchors(feat_h1, feat_w1,
                          self.img_h, self.img_w,
                          self.ANCHORS_S1).to(device)
        a2 = make_anchors(feat_h2, feat_w2,
                          self.img_h, self.img_w,
                          self.ANCHORS_S2).to(device)
        return a1, a2

    # ── Inference decode ──────────────────────────────────────────────────────

    @torch.no_grad()
    def decode_predictions(self,
                           preds_s1: torch.Tensor,
                           preds_s2: torch.Tensor,
                           conf_thresh: float = 0.3,
                           iou_thresh:  float = 0.4
                           ) :
        """
        Decode raw head outputs to bounding boxes for a batch.

        Returns list of dicts per image:
            {'boxes': [N,4] xyxy normalised, 'scores': [N]}
        """
        B  = preds_s1.shape[0]
        dev = preds_s1.device

        fh1, fw1 = preds_s1.shape[2], preds_s1.shape[3]
        fh2, fw2 = preds_s2.shape[2], preds_s2.shape[3]

        a1, a2 = self.get_anchors(fh1, fw1, fh2, fw2, device=dev)

        results = []
        for b in range(B):
            boxes_all  = []
            scores_all = []

            for preds, anchors in [(preds_s1[b], a1), (preds_s2[b], a2)]:
                # preds: [A, fh, fw, 5]
                fh, fw = preds.shape[1], preds.shape[2]
                p = preds.view(-1, 5)                   # [A*fh*fw, 5]

                confs = torch.sigmoid(p[:, 0])
                tx, ty, tw, th = p[:, 1], p[:, 2], p[:, 3], p[:, 4]

                # Decode offsets relative to anchors
                acx = anchors[:, 0]
                acy = anchors[:, 1]
                aw  = anchors[:, 2]
                ah  = anchors[:, 3]

                cx = torch.sigmoid(tx) * aw * 2 + acx - aw * 0.5
                cy = torch.sigmoid(ty) * ah * 2 + acy - ah * 0.5
                w  = aw  * (2 * torch.sigmoid(tw)) ** 2
                h  = ah  * (2 * torch.sigmoid(th)) ** 2

                x1 = torch.clamp(cx - w / 2, 0, 1)
                y1 = torch.clamp(cy - h / 2, 0, 1)
                x2 = torch.clamp(cx + w / 2, 0, 1)
                y2 = torch.clamp(cy + h / 2, 0, 1)

                keep = confs >= conf_thresh
                if keep.sum() == 0:
                    continue

                boxes_all.append(
                    torch.stack([x1[keep], y1[keep],
                                 x2[keep], y2[keep]], dim=1))
                scores_all.append(confs[keep])

            if not boxes_all:
                results.append({'boxes': torch.zeros(0, 4), 'scores': torch.zeros(0)})
                continue

            boxes  = torch.cat(boxes_all,  dim=0)
            scores = torch.cat(scores_all, dim=0)

            # NMS
            keep = _nms(boxes, scores, iou_thresh)
            results.append({'boxes': boxes[keep], 'scores': scores[keep]})

        return results


# ── NMS ───────────────────────────────────────────────────────────────────────

def _nms(boxes: torch.Tensor, scores: torch.Tensor,
         iou_thresh: float = 0.4) -> torch.Tensor:
    """Simple NMS. boxes: [N,4] xyxy normalised."""
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long)
    try:
        from torchvision.ops import nms
        return nms(boxes, scores, iou_thresh)
    except Exception:
        # Fallback: greedy NMS
        order  = scores.argsort(descending=True)
        keep   = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            ious = _box_iou(boxes[i:i+1], boxes[order[1:]])[0]
            order = order[1:][ious < iou_thresh]
        return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between boxes a [M,4] and b [N,4], returns [M,N]."""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    inter_x1 = torch.max(a[:, None, 0], b[None, :, 0])
    inter_y1 = torch.max(a[:, None, 1], b[None, :, 1])
    inter_x2 = torch.min(a[:, None, 2], b[None, :, 2])
    inter_y2 = torch.min(a[:, None, 3], b[None, :, 3])
    inter_w  = (inter_x2 - inter_x1).clamp(min=0)
    inter_h  = (inter_y2 - inter_y1).clamp(min=0)
    inter    = inter_w * inter_h
    union    = area_a[:, None] + area_b[None, :] - inter
    return inter / (union + 1e-6)
