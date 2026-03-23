#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import sys
import yaml

def load_modes(csv_path):
    modes = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["idx"])
            mode = row["mode"].strip()
            modes.append((idx, mode))
    return modes

def compress_segments(modes, keep=("LK", "LC", "INT")):
    """
    modes: list of (idx, mode)
    return: list of {name,start,end}
    - keep에 없는 mode(NONE 등)는 LK로 취급하고 싶으면 여기서 바꾸면 됨
    """
    segments = []
    if not modes:
        return segments

    # NONE은 LK로 합치고 싶으면 아래 한 줄:
    def norm(m):
        if m == "NONE" or m == "":
            return "LK"
        return m

    cur_name = norm(modes[0][1])
    cur_start = modes[0][0]
    prev_idx = modes[0][0]

    for idx, m in modes[1:]:
        m = norm(m)
        # idx가 연속이 아닐 수도 있으니, 인덱스 순서만 믿고 진행
        if m != cur_name:
            segments.append({"name": cur_name, "start": cur_start, "end": prev_idx})
            cur_name = m
            cur_start = idx
        prev_idx = idx

    segments.append({"name": cur_name, "start": cur_start, "end": prev_idx})

    # keep 필터(원하면 사용)
    # 예: LC/INT만 남기고 나머지는 LK로 병합하고 싶으면 로직 추가 가능
    return segments

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 make_segments_from_csv.py <path_with_mode.csv> <segments.yaml>")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_yaml = sys.argv[2]

    modes = load_modes(csv_path)
    segs = compress_segments(modes)

    data = {"segments": segs}

    with open(out_yaml, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"[OK] wrote {out_yaml}")
    print("segments preview:")
    for s in segs[:10]:
        print(s)

if __name__ == "__main__":
    main()