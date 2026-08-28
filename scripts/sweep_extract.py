import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.team_assigner import TeamAssigner  # noqa: E402

CLS_PLAYER = 2

# các cấu hình dải thân sẽ thử: (torso_top, torso_bot, center_ratio, nhãn)
CONFIGS = [
    (0.00, 0.50, 1.00, "hien tai: nua tren, full ngang"),
    (0.15, 0.45, 1.00, "bo dau (15-45%), full ngang"),
    (0.15, 0.45, 0.70, "bo dau + giua 70%"),
    (0.20, 0.45, 0.60, "nguc hep (20-45%) + giua 60%"),
    (0.10, 0.40, 0.80, "10-40% + giua 80%"),
    (0.20, 0.50, 0.50, "20-50% + giua 50%"),
    (0.25, 0.55, 0.60, "nguc thap (25-55%) + giua 60%"),
]


def score(colors, track_ids):
    """Chấm điểm một bộ màu đã trích: separation / ambiguous% / purity."""
    X = np.asarray(colors, np.float32)
    if len(X) < 10:
        return None
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    lab = km.labels_
    c0, c1 = km.cluster_centers_
    d_between = float(np.linalg.norm(c0 - c1))
    spread = float(np.mean([np.linalg.norm(x - km.cluster_centers_[l]) for x, l in zip(X, lab)]))
    separation = d_between / spread if spread > 1e-6 else 0.0
    d0 = np.linalg.norm(X - c0, axis=1)
    d1 = np.linalg.norm(X - c1, axis=1)
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)
    ambiguous = float(np.mean(near / np.maximum(far, 1e-6) > 0.8)) * 100
    per_track = defaultdict(list)
    for t, l in zip(track_ids, lab):
        per_track[t].append(int(l))
    purities = [Counter(v).most_common(1)[0][1] / len(v) for v in per_track.values() if len(v) >= 5]
    purity = float(np.mean(purities)) if purities else 0.0
    return {"separation": round(separation, 3), "ambiguous_pct": round(ambiguous, 1),
            "purity": round(purity, 4), "center_distance": round(d_between, 1),
            "n_samples": len(X), "n_tracks": len(per_track)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="output/sweep.json")
    ap.add_argument("--tracker", default="configs/bytetrack.yaml")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--per-track", type=int, default=40, dest="per_track",
                    help="số ảnh cắt (to nhất) giữ lại mỗi track")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(a.weights)
    cap = cv2.VideoCapture(a.source)
    cache = defaultdict(list)   # tid -> [(area, crop)]

    print("[1/2] Đang chạy tracking + cache ảnh cắt ...")
    results = model.track(source=a.source, tracker=a.tracker, persist=True, conf=a.conf,
                          imgsz=a.imgsz, device=a.device, stream=True, verbose=False)
    for r in results:
        ok, frame = cap.read()
        if not ok:
            break
        b = r.boxes
        if b is None or b.id is None:
            continue
        xyxy = b.xyxy.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        ids = b.id.cpu().numpy().astype(int)
        for bb, c, tid in zip(xyxy, cls, ids):
            if c != CLS_PLAYER:      # chỉ cầu thủ ngoài sân (GK/ref không dùng để phân đội)
                continue
            x1, y1, x2, y2 = map(int, bb)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 - x1 < 8 or y2 - y1 < 16:
                continue
            cache[int(tid)].append(((x2 - x1) * (y2 - y1), frame[y1:y2, x1:x2].copy()))
    cap.release()

    # chỉ giữ N ảnh TO NHẤT mỗi track (áo rõ nhất) để đỡ tốn RAM và đỡ nhiễu
    for tid in list(cache):
        cache[tid] = sorted(cache[tid], key=lambda z: -z[0])[: a.per_track]
    total = sum(len(v) for v in cache.values())
    print(f"      cache {len(cache)} track / {total} ảnh cắt")

    print("[2/2] Đang chấm điểm các cấu hình ...")
    rows = []
    for top, bot, cr, label in CONFIGS:
        ta = TeamAssigner(torso_top=top, torso_bot=bot, center_ratio=cr)
        cols, tids = [], []
        for tid, items in cache.items():
            for _, crop in items:
                c = ta.get_player_color(crop, [0, 0, crop.shape[1], crop.shape[0]])
                if c is not None:
                    cols.append(c)
                    tids.append(tid)
        s = score(cols, tids)
        if s:
            s.update({"torso_top": top, "torso_bot": bot, "center_ratio": cr, "label": label})
            rows.append(s)

    rows.sort(key=lambda z: (-z["purity"], -z["separation"], z["ambiguous_pct"]))
    print("\n%-34s %9s %10s %8s %8s" % ("CAU HINH", "purity", "separation", "ambig%", "d_tam"))
    for r in rows:
        print("%-34s %9.4f %10.3f %8.1f %8.1f" % (r["label"], r["purity"], r["separation"],
                                                  r["ambiguous_pct"], r["center_distance"]))
    best = rows[0]
    print(f"\n=> TOT NHAT: {best['label']}  "
          f"(torso_top={best['torso_top']}, torso_bot={best['torso_bot']}, "
          f"center_ratio={best['center_ratio']})")
    with open(a.out, "w") as f:
        json.dump({"results": rows, "best": best}, f, indent=1)
    print("đã ghi", a.out)


if __name__ == "__main__":
    main()