"""
explore_dataset.py — Diagnostic script to inspect the NEFER dataset structure.

Run this FIRST to understand your data before training.

Usage:
    python3 explore_dataset.py --raw_dir ~/Downloads/raw --frames_dir ~/Downloads/Project/event_frames
"""

import os
import sys
import argparse
import glob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw_dir',    default=None, help='Path to raw/ folder')
    p.add_argument('--frames_dir', default=None, help='Path to event_frames/ folder')
    return p.parse_args()


def explore_directory(path: str, label: str, depth: int = 3):
    print(f"\n{'='*60}")
    print(f"  {label}: {path}")
    print('='*60)
    if not os.path.isdir(path):
        print("  [NOT FOUND]")
        return

    for root, dirs, files in os.walk(path):
        dirs.sort()
        level = root.replace(path, '').count(os.sep)
        if level >= depth:
            del dirs[:]
            continue
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")
        sub = '  ' * (level + 1)
        for f in sorted(files)[:8]:
            size = os.path.getsize(os.path.join(root, f))
            print(f"{sub}{f}  ({size/1e3:.1f} KB)")
        if len(files) > 8:
            print(f"{sub}... and {len(files)-8} more files")


def check_raw_file(path: str):
    """Try to read the header of a .raw file."""
    try:
        with open(path, 'rb') as f:
            header_lines = []
            for _ in range(30):
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                try:
                    decoded = line.decode('latin-1').rstrip()
                    if decoded.startswith('%'):
                        header_lines.append(decoded)
                    else:
                        f.seek(pos)
                        break
                except Exception:
                    f.seek(pos)
                    break
            binary_start = f.tell()
            remaining = os.path.getsize(path) - binary_start
        return header_lines, remaining
    except Exception as e:
        return [], 0


def main():
    args = parse_args()

    print("\n" + "="*60)
    print("  NEFER Dataset Explorer")
    print("="*60)

    # ── Explore raw directory ──────────────────────────────────────────────────
    if args.raw_dir:
        explore_directory(args.raw_dir, "RAW events directory", depth=3)

        # Check a sample .raw file header
        raw_files = glob.glob(os.path.join(args.raw_dir, '**', '*.raw'),
                              recursive=True)
        print(f"\n  Total .raw files found: {len(raw_files)}")
        print(f"  Total .bias files found: "
              f"{len(glob.glob(os.path.join(args.raw_dir, '**', '*.bias'), recursive=True))}")

        if raw_files:
            sample = raw_files[0]
            print(f"\n  Inspecting sample: {os.path.basename(sample)}")
            header, n_binary = check_raw_file(sample)
            if header:
                print("  Header lines:")
                for h in header:
                    print(f"    {h}")
            else:
                print("  No ASCII header found (pure binary?)")
            print(f"  Binary data size: {n_binary/1e6:.2f} MB "
                  f"(~{n_binary//4:,} 4-byte words)")

    # ── Explore event_frames directory ────────────────────────────────────────
    if args.frames_dir:
        explore_directory(args.frames_dir, "event_frames directory", depth=4)

        # Look for annotation files
        ann_exts = ['*.json', '*.csv', '*.txt', '*.xml']
        print("\n  Searching for annotation files in event_frames ...")
        for ext in ann_exts:
            found = glob.glob(os.path.join(args.frames_dir, '**', ext),
                              recursive=True)
            if found:
                print(f"    {ext}: {len(found)} files")
                for f in found[:3]:
                    print(f"      → {f}")
                    # Show first few lines
                    try:
                        with open(f, 'r') as fh:
                            for i, line in enumerate(fh):
                                if i >= 3:
                                    break
                                print(f"         {line.rstrip()}")
                    except Exception:
                        pass

    # ── Summary & recommendations ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Summary & Next Steps")
    print("="*60)

    if args.raw_dir:
        raw_files = glob.glob(os.path.join(args.raw_dir, '**', '*.raw'), recursive=True)
        users = set(os.path.basename(os.path.dirname(f)) for f in raw_files)
        print(f"\n  ✓ Found {len(raw_files)} .raw files across {len(users)} users")

    if args.frames_dir:
        ann_files = []
        for ext in ['*.json', '*.csv', '*.txt']:
            ann_files += glob.glob(os.path.join(args.frames_dir, '**', ext),
                                   recursive=True)
        if ann_files:
            print(f"  ✓ Found {len(ann_files)} potential annotation files in event_frames/")
        else:
            print("\n  ⚠ No annotation files found in event_frames/")
            print("    Face bounding-box annotations may need to be downloaded")
            print("    separately from: https://github.com/miccunifi/NEFER")
            print("    Look for files like 'annotations.json' or 'bbox.csv' in the repo.")

    print()


if __name__ == '__main__':
    main()
