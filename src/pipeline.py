import argparse
import json
from collections import Counter, defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

from .team_assigner import TeamAssigner

# id lớp theo data.yaml: ['ball','goalkeeper','player','referee']
CLS_BALL, CLS_GK, CLS_PLAYER, CLS_REF = 0, 1, 2, 3

TEAM_BGR = {0: (0, 0, 235), 1: (235, 120, 0)}   # đội 0 đỏ, đội 1 xanh dương
REF_BGR = (0, 235, 235)
BALL_BGR = (255, 255, 255)
NEUTRAL = (200, 200, 200)


def draw_box(img, xyxy, color, label):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def role_of(cls_id):
    return {CLS_BALL: "ball", CLS_GK: "gk", CLS_PLAYER: "player", CLS_REF: "referee"}[cls_id]


def run(weights, source, out_path, json_path,
        tracker="bytetrack.yaml", conf=0.3, imgsz=640, fit_window=30,
        min_track_len=8, gk_border=0.15, min_coverage=0.30, vote_window=25, device=None):
    model = YOLO(weights)
    assigner = TeamAssigner()
    fitted = False
    raw = defaultdict(list)   # track_id -> [{frame, cls, bbox, cx, cy}]

    # ---------------- LƯỢT 1: track + bỏ phiếu đội ----------------
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    results = model.track(source=source, tracker=tracker, persist=True, conf=conf,
                          imgsz=imgsz, device=device, stream=True, verbose=False)
    frame_idx = 0
    for r in results:
        ok, frame = cap.read()
        if not ok:
            break
        b = r.boxes
        if b is not None and b.id is not None:
            xyxy = b.xyxy.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
            ids = b.id.cpu().numpy().astype(int)
            # chỉ cầu thủ ngoài sân để fit đội (loại GK khỏi fit)
            outfield = [bb for bb, c in zip(xyxy, cls) if c == CLS_PLAYER]
            if not fitted:
                assigner.collect(frame, outfield)
                if frame_idx >= fit_window:
                    fitted = assigner.fit_from_buffer()
            for bb, c, tid in zip(xyxy, cls, ids):
                x1, y1, x2, y2 = map(float, bb)
                rec = {"frame": frame_idx, "cls": int(c), "bbox": [x1, y1, x2, y2],
                       "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "vote": None, "w": 0.0}
                if c in (CLS_PLAYER, CLS_GK) and fitted:
                    t_i, w_i = assigner.classify(frame, bb)   # KHÔNG tích luỹ
                    rec["vote"], rec["w"] = t_i, w_i
                raw[int(tid)].append(rec)
        frame_idx += 1
    cap.release()
    n_frames = frame_idx

    # ---------------- HẬU XỬ LÝ ----------------
    def main_cls(recs):
        return Counter(r["cls"] for r in recs).most_common(1)[0][0]

    # min_coverage: track chỉ bỏ phiếu được ở rất ít frame (áo lạ như GK) thì màu
    # KHÔNG đáng tin -> coi như chưa có đội, nhường cho luật GK-theo-vị-trí.
    # BẦU ĐỘI THEO CỬA SỔ TRƯỢT (thay cho "một đội cho cả track").
    # Lý do (đã đo trên clip thật): máy quay lia nhanh khiến ByteTrack HOÁN ID sang
    # người khác giữa clip. Chốt một đội cho cả track làm nửa clip sai vĩnh viễn.
    # Cửa sổ trượt cho nhãn bám theo áo thật trong ~1 giây.
    per_det_team = {}          # (tid, frame) -> team|None
    for tid, recs in raw.items():
        if main_cls(recs) not in (CLS_PLAYER, CLS_GK):
            continue
        n = len(recs)
        for i, r in enumerate(recs):
            lo, hi = max(0, i - vote_window), min(n, i + vote_window + 1)
            c = Counter()
            for j in range(lo, hi):
                if recs[j]["vote"] is not None:
                    c[recs[j]["vote"]] += recs[j]["w"]
            per_det_team[(tid, r["frame"])] = c.most_common(1)[0][0] if c else None

    # LÀM MƯỢT LỚP THEO THỜI GIAN: vai trò (player/gk/referee) của một track không
    # nên nhảy từng frame. Đo trên clip thật: 11/62 track bị nhấp nháy player<->referee
    # (cao nhất 45% số frame) do model gặp màu áo lạ. Lấy đa số trong cửa sổ trượt.
    per_det_cls = {}
    for tid, recs in raw.items():
        n = len(recs)
        for i, r in enumerate(recs):
            lo, hi = max(0, i - vote_window), min(n, i + vote_window + 1)
            c = Counter(recs[j]["cls"] for j in range(lo, hi))
            per_det_cls[(tid, r["frame"])] = c.most_common(1)[0][0]

    # đội "đại diện" của track (dùng cho thống kê nửa sân + luật GK)
    final_team = {}
    for tid, recs in raw.items():
        if main_cls(recs) not in (CLS_PLAYER, CLS_GK):
            final_team[tid] = None
            continue
        cov = sum(1 for r in recs if r["vote"] is not None) / max(1, len(recs))
        if cov < min_coverage:
            final_team[tid] = None      # màu không đáng tin (áo lạ) -> nhường luật GK
            continue
        c = Counter()
        for r in recs:
            if r["vote"] is not None:
                c[r["vote"]] += r["w"]
        final_team[tid] = c.most_common(1)[0][0] if c else None

    # đội nào trấn giữ nửa sân nào (từ cx trung bình của cầu thủ ngoài sân đã có đội)
    team_cx = defaultdict(list)
    for tid, recs in raw.items():
        t = final_team[tid]
        if t is not None and main_cls(recs) == CLS_PLAYER:
            team_cx[t].extend(r["cx"] for r in recs)
    left_team = right_team = None
    if len(team_cx) == 2:
        means = {t: float(np.mean(v)) for t, v in team_cx.items()}
        left_team = min(means, key=means.get)
        right_team = max(means, key=means.get)

    # GÁN GK THEO VỊ TRÍ: track chưa có đội (áo lạ) + nằm sát biên trái/phải.
    # Ghi đè LUÔN cả vai trò -> "gk" (điều kiện chặt để không nhầm cầu thủ biên):
    #   (a) áo lạ (team=None, không thuộc 2 cụm đội)  (b) sát biên trái/phải
    #   (c) sống đủ lâu (GK có mặt gần cả clip, khác cầu thủ chạy ngang qua biên).
    # Một track được coi là GK nếu:
    #   (a) model đã nhận là lớp goalkeeper, HOẶC
    #   (b) màu áo là NGOẠI LAI (không thuộc 2 cụm đội) và sống đủ lâu.
    # Đội của GK = đội đang trấn giữ NỬA SÂN chứa GK (không dùng ngưỡng biên cứng,
    # vì GK có thể đứng cách biên khá xa - audit thấy cx=1604/1920 = 83%).
    gk_assigned = 0
    forced_gk = set()
    gk_min_life = max(min_track_len, int(0.25 * n_frames))
    for tid, recs in raw.items():
        mc = main_cls(recs)
        if mc not in (CLS_PLAYER, CLS_GK):
            continue
        is_model_gk = (mc == CLS_GK)
        is_color_outlier = (final_team[tid] is None)
        if not (is_model_gk or (is_color_outlier and len(recs) >= gk_min_life)):
            continue
        if len(recs) < min_track_len:
            continue
        mean_cx = float(np.mean([r["cx"] for r in recs]))
        half_team = left_team if mean_cx < W / 2 else right_team
        if half_team is None:
            continue
        final_team[tid] = half_team
        forced_gk.add(tid)
        gk_assigned += 1

    # LỌC TRACK CHẾT YỂU (giảm ID rác lúc tranh chấp), giữ nguyên bóng
    keep = {tid for tid, recs in raw.items()
            if main_cls(recs) == CLS_BALL or len(recs) >= min_track_len}

    # ---------------- LƯỢT 2: vẽ lại ----------------
    per_frame = defaultdict(list)
    for tid, recs in raw.items():
        if tid in keep:
            for r in recs:
                per_frame[r["frame"]].append((tid, r))

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    cap = cv2.VideoCapture(source)
    fi = 0
    while fi < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        for tid, r in per_frame.get(fi, []):
            c = per_det_cls.get((tid, r["frame"]), r["cls"]); bb = r["bbox"]
            if c == CLS_BALL:
                draw_box(frame, bb, BALL_BGR, "ball")
            elif c == CLS_REF:
                draw_box(frame, bb, REF_BGR, f"ref {tid}")
            else:
                team = (final_team[tid] if tid in forced_gk
                        else per_det_team.get((tid, r["frame"]), final_team[tid]))
                color = TEAM_BGR.get(team, NEUTRAL)
                role = "gk" if (c == CLS_GK or tid in forced_gk) else "player"
                lbl = f"{role} {tid}" + (f" T{team}" if team is not None else "")
                draw_box(frame, bb, color, lbl)
        writer.write(frame)
        fi += 1
    cap.release()
    writer.release()

    # ---------------- xuất tracks.json ----------------
    tracks_out = defaultdict(list)
    for tid, recs in raw.items():
        if tid not in keep:
            continue
        for r in recs:
            role = role_of(per_det_cls.get((tid, r["frame"]), r["cls"]))
            if tid in forced_gk:
                role = "gk"
            if role not in ("player", "gk"):
                team = None
            elif tid in forced_gk:
                team = final_team[tid]
            else:
                team = per_det_team.get((tid, r["frame"]), final_team[tid])
            tracks_out[int(tid)].append({"frame": r["frame"], "role": role, "team": team,
                                         "bbox": r["bbox"], "cx": r["cx"], "cy": r["cy"]})
    meta = {
        "fps": fps, "width": W, "height": H, "frames": n_frames,
        "n_tracks": len(tracks_out),
        "team_colors_bgr": {str(k): [float(x) for x in v] for k, v in assigner.team_colors.items()},
        "defends_left_team": left_team, "defends_right_team": right_team,
        "gk_assigned_by_position": gk_assigned,
        "tracks": tracks_out,
    }
    with open(json_path, "w") as f:
        json.dump(meta, f)
    print(f"[done] {n_frames} frames | kept {len(tracks_out)}/{len(raw)} IDs "
          f"| GK gán theo vị trí: {gk_assigned} | video->{out_path} | json->{json_path}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--json", default="tracks.json")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-track-len", type=int, default=8, dest="min_track_len")
    ap.add_argument("--gk-border", type=float, default=0.15, dest="gk_border")
    ap.add_argument("--vote-window", type=int, default=25, dest="vote_window",
                    help="nửa cửa sổ (frame) bầu đội; 25 ~ 1s mỗi bên @25fps")
    ap.add_argument("--min-coverage", type=float, default=0.30, dest="min_coverage",
                    help="tỉ lệ frame tối thiểu bỏ được phiếu màu thì mới nhận đội")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    run(a.weights, a.source, a.out, a.json, tracker=a.tracker, conf=a.conf,
        imgsz=a.imgsz, min_track_len=a.min_track_len, gk_border=a.gk_border,
        min_coverage=a.min_coverage, vote_window=a.vote_window, device=a.device)