from __future__ import annotations
"""
train.py — Main Training Script for NEFER Event-Based Face Detection

Usage:
    python train.py --data_dir /path/to/event_raw --epochs 150 --batch_size 32

The script trains from random weights (no transfer learning).
Best checkpoint is saved to checkpoints/best_model.pth
Final submission model is saved to final_model.pth
"""

import os
import sys
import time
import argparse
import json
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from dataset  import NEFERDataset
from model    import EventFaceDetector
from loss     import DetectionLoss
from evaluate import evaluate, print_metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='NEFER Event Face Detector')
    p.add_argument('--data_dir',    type=str,   required=True,
                   help='Path to raw/ directory (contains user_XX subfolders with .raw files)')
    p.add_argument('--ann_dir',     type=str,   default=None,
                   help='Path to annotation folder (bbox JSON/CSV/TXT files). '
                        'Auto-detected if not set.')
    p.add_argument('--frames_dir',  type=str,   default=None,
                   help='Path to event_frames/ folder (used to find annotations)')
    p.add_argument('--out_dir',     type=str,   default='checkpoints',
                   help='Directory for saved models & logs')
    p.add_argument('--epochs',      type=int,   default=150)
    p.add_argument('--batch_size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=3e-3,
                   help='Peak learning rate for OneCycleLR')
    p.add_argument('--weight_decay',type=float, default=1e-4)
    p.add_argument('--window_ms',   type=float, default=50.0,
                   help='Event window duration in ms')
    p.add_argument('--T',           type=int,   default=5,
                   help='Voxel grid time bins')
    p.add_argument('--img_h',       type=int,   default=720)
    p.add_argument('--img_w',       type=int,   default=1280)
    p.add_argument('--min_events',  type=int,   default=50,
                   help='Skip windows with fewer events (sparse robustness)')
    p.add_argument('--num_workers', type=int,   default=4)
    p.add_argument('--lambda_box',  type=float, default=5.0)
    p.add_argument('--seed',        type=int,   default=42)
    p.add_argument('--device',      type=str,   default='auto')
    p.add_argument('--resume',      type=str,   default=None,
                   help='Path to checkpoint to resume from')
    p.add_argument('--val_every',   type=int,   default=5,
                   help='Validate every N epochs')
    p.add_argument('--warmup_pct',  type=float, default=0.1,
                   help='Fraction of steps used for LR warm-up')
    return p.parse_args()


# ── Seeding ────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Training Step ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler,
                    criterion, anchors_s1, anchors_s2,
                    device, img_h, img_w, epoch, scaler=None):
    model.train()
    total_loss = conf_loss_sum = box_loss_sum = 0.0
    n_batches  = len(loader)

    for batch_idx, (voxel, gt_boxes) in enumerate(loader):
        voxel    = voxel.to(device, non_blocking=True)
        gt_boxes = gt_boxes.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                preds_s1, preds_s2 = model(voxel)
                losses = criterion(preds_s1, preds_s2,
                                   anchors_s1, anchors_s2,
                                   gt_boxes, img_h, img_w)
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds_s1, preds_s2 = model(voxel)
            losses = criterion(preds_s1, preds_s2,
                               anchors_s1, anchors_s2,
                               gt_boxes, img_h, img_w)
            losses['total'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        total_loss    += losses['total'].item()
        conf_loss_sum += losses['conf_loss'].item()
        box_loss_sum  += losses['box_loss'].item()

        if batch_idx % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  [ep{epoch:03d} {batch_idx+1:4d}/{n_batches}] "
                  f"loss={losses['total'].item():.4f}  "
                  f"conf={losses['conf_loss'].item():.4f}  "
                  f"box={losses['box_loss'].item():.4f}  "
                  f"lr={lr:.2e}")

    n = max(n_batches, 1)
    return {
        'train_loss'     : total_loss    / n,
        'train_conf_loss': conf_loss_sum / n,
        'train_box_loss' : box_loss_sum  / n,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else \
                 'mps'  if torch.backends.mps.is_available() else 'cpu'
    else:
        device = args.device
    print(f"[train] Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("[train] Building datasets …")
    train_ds = NEFERDataset(
        raw_dir    = args.data_dir,
        ann_dir    = args.ann_dir,
        split      = 'train',
        window_ms  = args.window_ms,
        T          = args.T,
        height     = args.img_h,
        width      = args.img_w,
        min_events = args.min_events,
        augment    = True,
        seed       = args.seed,
    )
    val_ds = NEFERDataset(
        raw_dir    = args.data_dir,
        ann_dir    = args.ann_dir,
        split      = 'val',
        window_ms  = args.window_ms,
        T          = args.T,
        height     = args.img_h,
        width      = args.img_w,
        min_events = args.min_events,
        augment    = False,
        seed       = args.seed,
    )

    if len(train_ds) == 0:
        sys.exit("[ERROR] Training set is empty.\n"
                 "  1. Check --data_dir points to the raw/ folder\n"
                 "  2. Run: python3 explore_dataset.py --raw_dir <raw_dir> --frames_dir <event_frames>\n"
                 "  3. Add --ann_dir pointing to your annotation folder")

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = (device == 'cuda'),
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = args.batch_size * 2,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = (device == 'cuda'),
    )

    print(f"[train] train={len(train_ds)} samples  val={len(val_ds)} samples")

    # ── Model ────────────────────────────────────────────────────────────────
    in_channels = args.T * 2
    model = EventFaceDetector(
        in_channels = in_channels,
        img_h       = args.img_h,
        img_w       = args.img_w,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Model params: {n_params:,}  (~{n_params/1e6:.2f}M)")

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(),
                      lr           = args.lr,
                      weight_decay = args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr          = args.lr,
        total_steps     = total_steps,
        pct_start       = args.warmup_pct,
        anneal_strategy = 'cos',
        div_factor      = 25.0,
        final_div_factor= 1e4,
    )

    criterion = DetectionLoss(lambda_box=args.lambda_box).to(device)

    # AMP scaler (CUDA only)
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_ap50     = 0.0
    history       = []

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_ap50   = ckpt.get('best_ap50', 0.0)
        history     = ckpt.get('history', [])
        print(f"[train] Resumed from epoch {start_epoch-1}  "
              f"best_ap50={best_ap50:.3f}")

    # ── Pre-compute anchors (fixed for all batches) ───────────────────────────
    # We need a dummy forward to know feature-map sizes
    dummy = torch.zeros(1, in_channels, args.img_h, args.img_w).to(device)
    with torch.no_grad():
        ps1, ps2 = model(dummy)
    fh1, fw1 = ps1.shape[2], ps1.shape[3]
    fh2, fw2 = ps2.shape[2], ps2.shape[3]
    anchors_s1, anchors_s2 = model.get_anchors(
        fh1, fw1, fh2, fw2, device=device
    )
    print(f"[train] Feature maps: P4={fh1}×{fw1} ({fh1*fw1*3} anchors)  "
          f"P5={fh2}×{fw2} ({fh2*fw2*3} anchors)")

    # ── Training Loop ─────────────────────────────────────────────────────────
    print(f"\n[train] Starting training for {args.epochs} epochs …\n")
    t_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        ep_t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, anchors_s1, anchors_s2,
            device, args.img_h, args.img_w, epoch, scaler
        )

        ep_time = time.time() - ep_t0
        print(f"[ep{epoch:03d}] train_loss={train_metrics['train_loss']:.4f}  "
              f"time={ep_time:.1f}s")

        # ── Validation ───────────────────────────────────────────────────────
        val_metrics = {}
        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(
                model, val_loader, device,
                conf_thresh = 0.3,
                iou_thresh  = 0.5,
                max_batches = None,
            )
            print_metrics(val_metrics, epoch=epoch)

            ap50 = val_metrics.get('ap50', 0.0)
            if ap50 > best_ap50:
                best_ap50 = ap50
                _save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    best_ap50, history, args,
                    os.path.join(args.out_dir, 'best_model.pth')
                )
                print(f"  ✓ New best AP@50={best_ap50:.3f} — checkpoint saved")

        # Periodic checkpoint
        if epoch % 25 == 0:
            _save_checkpoint(
                model, optimizer, scheduler, epoch,
                best_ap50, history, args,
                os.path.join(args.out_dir, f'ckpt_ep{epoch:03d}.pth')
            )

        # Log history
        record = {'epoch': epoch, **train_metrics, **val_metrics}
        history.append(record)
        _save_json(history, os.path.join(args.out_dir, 'history.json'))

    # ── Final export ─────────────────────────────────────────────────────────
    total_time = (time.time() - t_start) / 60
    print(f"\n[train] Done. Total time: {total_time:.1f} min  "
          f"Best AP@50: {best_ap50:.3f}")

    # Load best weights and save final submission model
    best_path = os.path.join(args.out_dir, 'best_model.pth')
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print("[train] Loaded best checkpoint for final export")

    # Full PyTorch save (state dict + config)
    final_save = {
        'model_state_dict': model.state_dict(),
        'config': {
            'in_channels': in_channels,
            'img_h'      : args.img_h,
            'img_w'      : args.img_w,
            'T'          : args.T,
            'window_ms'  : args.window_ms,
        },
        'architecture': 'EventFaceDetector',
        'best_ap50'   : best_ap50,
    }
    torch.save(final_save, 'final_model.pth')
    print("[train] Saved → final_model.pth")

    # TorchScript export for fast inference
    model.eval()
    try:
        scripted = torch.jit.script(model)
        scripted.save('final_model_scripted.pt')
        print("[train] Saved → final_model_scripted.pt (TorchScript)")
    except Exception as e:
        print(f"[warn] TorchScript export failed: {e}")
        print("       (final_model.pth still valid for submission)")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save_checkpoint(model, optimizer, scheduler, epoch,
                     best_ap50, history, args, path):
    torch.save({
        'epoch'               : epoch,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_ap50'           : best_ap50,
        'history'             : history,
        'config': {
            'in_channels': args.T * 2,
            'img_h'      : args.img_h,
            'img_w'      : args.img_w,
        },
    }, path)


def _save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


if __name__ == '__main__':
    main()
