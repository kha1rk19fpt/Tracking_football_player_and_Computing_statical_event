import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.team_assigner import TeamAssigner  # noqa: E402

CLS_BALL, CLS_GK, CLS_PLAYER, CLS_REF = 0, 1, 2, 3
ROLE = {CLS_BALL: "ball", CLS_GK: "gk", CLS_PLAYER: "player", CLS_REF: "referee"}

CROP_W, CROP_H = 50, 80
N_CROPS = 5
LABEL_W, SWATCH_W = 250, 55
ROW_H = CROP_H + 6
MAX_ROWS_PER_SHEET = 22


def build_row(label_lines, crops, jersey_bgr, team_bgr):
    row = np.full((ROW_H, LABEL_W + N_CROPS * (CROP_W + 4) + 2 * (SWATCH_W + 4), 3), 30, np.uint8)
    for i, ln in enumerate(label_lines):
        cv2.putText(row, ln, (6, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (235, 235, 235), 1, cv2.LINE_AA)
    x = LABEL_W
    for i in range(N_CROPS):
        if i < len(crops):
            c = cv2.resize(crops[i], (CROP_W, CROP_H))
            row[3:3 + CROP_H, x:x + CROP_W] = c
        x += CROP_W + 4
    if jersey_bgr is not None:
        row[3:3 + CROP_H, x:x + SWATCH_W] = np.array(jersey_bgr, np.uint8)
        cv2.putText(row, "trich", (x + 4, 3 + CROP_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 0, 0), 1, cv2.LINE_AA)
    x += SWATCH_W + 4
    if team_bgr is not None:
        row[3:3 + CROP_H, x:x + SWATCH_W] = np.array(team_bgr, np.uint8)
        cv2.putText(row, "doi", (x + 4, 3 + CROP_H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--outdir", default="output/audit")
    ap.add_argument("--tracker", default="configs/bytetrack.yaml")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--fit-window", type=int, default=30, dest="fit_window")
    ap.add_argument("--min-frames", type=int, default=30, dest="min_frames",
                    help="chỉ audit track sống >= N frame")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(a.weights)
    assigner = TeamAssigner()
    fitted = False

    cap = cv2.VideoCapture(a.source)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    samples = defaultdict(list)   # tid -> [(area, frame_idx, crop, jersey_color)]
    meta = defaultdict(lambda: {"cls": Counter(), "cx": [], "conf": defaultdict(list)})

    results = model.track(source=a.source, tracker=a.tracker, persist=True, conf=a.conf,
                          imgsz=a.imgsz, device=a.device, stream=True, verbose=False)
    fi = 0
    for r in results:
        ok, frame = cap.read()
        if not ok:
            break
        b = r.boxes
        if b is not None and b.id is not None:
            xyxy = b.xyxy.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
            ids = b.id.cpu().numpy().astype(int)
            confs = (b.conf.cpu().numpy() if b.conf is not None
                     else np.ones(len(cls), np.float32))
            outfield = [bb for bb, c in zip(xyxy, cls) if c == CLS_PLAYER]
            if not fitted:
                assigner.collect(frame, outfield)
                if fi >= a.fit_window:
                    fitted = assigner.fit_from_buffer()
            for bb, c, tid, cf in zip(xyxy, cls, ids, confs):
                tid = int(tid)
                meta[tid]["cls"][int(c)] += 1
                meta[tid].setdefault("conf", defaultdict(list))[int(c)].append(float(cf))
                x1, y1, x2, y2 = map(int, bb)
                meta[tid]["cx"].append((x1 + x2) / 2)
                if c in (CLS_PLAYER, CLS_GK):
                    if fitted:
                        assigner.assign(frame, bb, tid)
                    col = assigner.get_player_color(frame, bb)
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    if x2c - x1c > 4 and y2c - y1c > 4:
                        area = (x2c - x1c) * (y2c - y1c)
                        samples[tid].append((area, fi, frame[y1c:y2c, x1c:x2c].copy(),
                                             None if col is None else col.copy()))
        fi += 1
    cap.release()

    if not fitted:
        print("LỖI: chưa fit được cụm đội (clip quá ngắn hoặc không thấy cầu thủ).")
        return

    c0 = assigner.team_colors[0]
    c1 = assigner.team_colors[1]
    diag = {"team_colors_bgr": {"0": [float(x) for x in c0], "1": [float(x) for x in c1]},
            "center_distance": float(np.linalg.norm(c0 - c1)),
            "feature_space": assigner.feature_space,
            "outlier_thresh": float(assigner._outlier_thresh), "tracks": {}}

    rows = []
    for tid, s in sorted(samples.items(), key=lambda kv: -len(kv[1])):
        if len(s) < a.min_frames:
            continue
        main_c = meta[tid]["cls"].most_common(1)[0][0]
        votes = assigner._track_votes.get(tid, Counter())
        tot = sum(votes.values())
        team = votes.most_common(1)[0][0] if tot else None
        margin = (votes[team] / tot) if tot else 0.0
        cols = [x[3] for x in s if x[3] is not None]
        mean_col = np.mean(cols, axis=0) if cols else None
        d0 = float(np.linalg.norm(mean_col - c0)) if mean_col is not None else None
        d1 = float(np.linalg.norm(mean_col - c1)) if mean_col is not None else None
        diag["tracks"][str(tid)] = {
            "role": ROLE[main_c], "team": team, "n_frames": len(s),
            "vote_margin": round(margin, 3),
            "votes": {str(k): float(v) for k, v in votes.items()},
            "mean_jersey_bgr": None if mean_col is None else [round(float(x), 1) for x in mean_col],
            "dist_to_T0": None if d0 is None else round(d0, 1),
            "dist_to_T1": None if d1 is None else round(d1, 1),
            "mean_cx": round(float(np.mean(meta[tid]["cx"])), 1), "frame_width": W,
            "cls_counts": {ROLE[k]: v for k, v in meta[tid]["cls"].items()},
            "mean_conf_by_cls": {ROLE[k]: round(float(np.mean(v)), 3)
                                 for k, v in meta[tid]["conf"].items()},
        }
        # lấy N ảnh cắt TO NHẤT (áo rõ nhất) trải theo thời gian
        big = sorted(s, key=lambda x: -x[0])[: N_CROPS * 4]
        big = sorted(big, key=lambda x: x[1])
        step = max(1, len(big) // N_CROPS)
        crops = [big[i][2] for i in range(0, len(big), step)][:N_CROPS]
        lines = [f"id{tid} {ROLE[main_c]} T{team} m={margin:.2f}",
                 f"f={len(s)} cx={int(np.mean(meta[tid]['cx']))}",
                 f"d0={0 if d0 is None else int(d0)} d1={0 if d1 is None else int(d1)}"]
        rows.append(build_row(lines, crops,
                              None if mean_col is None else mean_col,
                              c0 if team == 0 else (c1 if team == 1 else None)))

    hdr = build_row(["TAM CUM DOI:", f"T0 va T1, cach nhau {diag['center_distance']:.0f}",
                     f"khong gian={assigner.feature_space}"], [], c0, c1)
    sheets = 0
    for i in range(0, len(rows), MAX_ROWS_PER_SHEET):
        chunk = [hdr] + rows[i:i + MAX_ROWS_PER_SHEET]
        sheet = np.vstack(chunk)
        p = os.path.join(a.outdir, f"audit_sheet_{sheets + 1}.jpg")
        cv2.imwrite(p, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print("đã ghi", p, sheet.shape)
        sheets += 1
    with open(os.path.join(a.outdir, "audit.json"), "w") as f:
        json.dump(diag, f, indent=1)
    print("đã ghi", os.path.join(a.outdir, "audit.json"),
          f"| {len(rows)} track được audit | tâm cụm cách nhau {diag['center_distance']:.1f}")


if __name__ == "__main__":
    main()