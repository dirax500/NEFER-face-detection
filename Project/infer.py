"""
infer.py — Run inference on raw event files using the trained model.

Usage:
    python infer.py --model final_model.pth --events path/to/events.txt

Output: detected bounding boxes printed to stdout and optionally saved.
"""

import argparse
import time
import numpy as np
import torch

from dataset import parse_events, events_to_voxel_grid
from model   import EventFaceDetector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',       required=True, help='Path to final_model.pth')
    p.add_argument('--events',      required=True, help='Path to raw events .txt')
    p.add_argument('--window_ms',   type=float, default=50.0)
    p.add_argument('--conf_thresh', type=float, default=0.35)
    p.add_argument('--device',      default='auto')
    return p.parse_args()


def load_model(model_path: str, device: str) :
    ckpt = torch.load(model_path, map_location=device)
    cfg  = ckpt['config']
    model = EventFaceDetector(
        in_channels = cfg['in_channels'],
        img_h       = cfg['img_h'],
        img_w       = cfg['img_w'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    print(f"[infer] Model loaded from {model_path}  "
          f"(AP@50={ckpt.get('best_ap50', '?'):.3f})")
    return model, cfg


@torch.no_grad()
def detect_from_events(model, events: np.ndarray,
                       cfg: dict, t_ref: float,
                       window_us: float,
                       device: str,
                       conf_thresh: float = 0.35
                       ) :
    """
    Run detection on a single time window of events.

    Args:
        events    : full event array [N, 4]
        t_ref     : reference timestamp (µs) — end of window
        window_us : window duration in µs

    Returns:
        list of dicts: {'box_xyxy_norm': [4], 'score': float}
    """
    mask   = (events[:, 0] >= t_ref - window_us) & (events[:, 0] <= t_ref)
    win_ev = events[mask]

    voxel  = events_to_voxel_grid(win_ev, T=cfg['T'],
                                  height=cfg['img_h'],
                                  width=cfg['img_w'])
    voxel  = voxel.unsqueeze(0).to(device)   # [1, C, H, W]

    t0 = time.perf_counter()
    ps1, ps2 = model(voxel)
    if device == 'cuda':
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1000

    decoded = model.decode_predictions(ps1, ps2,
                                       conf_thresh=conf_thresh,
                                       iou_thresh=0.4)
    result = decoded[0]
    detections = []
    for i in range(result['boxes'].shape[0]):
        detections.append({
            'box_xyxy_norm': result['boxes'][i].cpu().tolist(),
            'score'        : result['scores'][i].item(),
        })
    return detections, latency_ms


def main():
    args = parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else \
                 'mps'  if torch.backends.mps.is_available() else 'cpu'
    else:
        device = args.device

    model, cfg = load_model(args.model, device)
    events     = parse_events(args.events)

    print(f"[infer] Loaded {len(events):,} events from {args.events}")
    print(f"[infer] Time range: {events[0,0]:.0f}µs → {events[-1,0]:.0f}µs")

    window_us = args.window_ms * 1_000.0

    # Slide window every 25ms across the full event stream
    t_start = events[0,  0] + window_us
    t_end   = events[-1, 0]
    stride  = 25_000.0  # 25ms

    t = t_start
    total_latency = []
    while t <= t_end:
        dets, lat = detect_from_events(
            model, events, cfg, t, window_us, device, args.conf_thresh
        )
        total_latency.append(lat)
        for d in dets:
            b = d['box_xyxy_norm']
            print(f"t={t/1e6:.3f}s  score={d['score']:.3f}  "
                  f"box=[{b[0]:.3f},{b[1]:.3f},{b[2]:.3f},{b[3]:.3f}]")
        t += stride

    if total_latency:
        avg = sum(total_latency) / len(total_latency)
        print(f"\n[infer] Avg latency per frame: {avg:.2f} ms  "
              f"({1000/avg:.1f} FPS)")


if __name__ == '__main__':
    main()
