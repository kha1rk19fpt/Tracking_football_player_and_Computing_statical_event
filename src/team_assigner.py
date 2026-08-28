from collections import Counter, defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self, torso_ratio: float = 0.5, center_ratio: float = 0.80,
                 outlier_factor: float = 1.10, v_weight: float = 0.35,
                 dark_v_floor: float = 50.0, feature_space: str = "bgr",
                 torso_top: float = 0.10, torso_bot: float | None = 0.40,
                 center_ratio_default_note: None = None):
        # MẶC ĐỊNH (0.10, 0.40, center_ratio=0.80) được chọn bằng QUÉT THAM SỐ trên
        # clip thật (scripts/sweep_extract.py): purity 0.9841 vs 0.9780 của bản cũ.
        # Lưu ý đã đo được: cắt SÁT ngực hơn (0.20-0.45) cho màu áo thuần hơn nhiều
        # (khoảng cách tâm cụm 168 vs 98) NHƯNG purity tụt còn 0.89 — vì vùng lấy màu
        # quá nhỏ với cầu thủ ở xa nên màu nhiễu. Đừng siết thêm.
        # torso_top/torso_bot: dải thân (theo tỉ lệ chiều cao bbox) dùng lấy màu áo.
        #   vd (0.15, 0.45) = chỉ lấy NGỰC, bỏ đầu (da/tóc) và quần -> ít lẫn tạp hơn.
        self.torso_top = torso_top
        self.torso_bot = torso_bot
        self._vote_frames: dict[int, int] = defaultdict(int)   # số frame BỎ ĐƯỢC phiếu
        self._seen_frames: dict[int, int] = defaultdict(int)   # số frame track xuất hiện
        # feature_space: "bgr" = phân cụm trên màu BGR thô (ĐÃ KIỂM CHỨNG, mặc định).
        #   "hsv" = đặc trưng HSV hạ trọng số V — THỬ NGHIỆM: trên clip broadcast nó cho
        #   ra ĐÚNG CÙNG phân hoạch như bgr (không lợi) nhưng làm áo đen (GK) lọt cửa
        #   ngoại lai -> hỏng luật gán GK theo vị trí. Chỉ bật khi có bằng chứng mới.
        self.feature_space = feature_space
        # torso_ratio : lấy bao nhiêu phần TRÊN của bbox làm vùng áo (0.5 = nửa trên)
        # center_ratio: lấy bao nhiêu phần GIỮA theo chiều ngang. MẶC ĐỊNH 1.0 = GIỮ
        #               NGUYÊN bề ngang để 4 góc còn viền cỏ cho dò-nền.
        # outlier_factor: ngưỡng ngoại lai = outlier_factor * (nửa khoảng cách 2 tâm cụm).
        # v_weight: trọng số kênh V (độ sáng) khi phân cụm. <1 để giảm ảnh hưởng
        #           bóng đổ/ánh sáng (V nhiễu nhất); S và H mới là dấu hiệu đội ổn định.
        self.torso_ratio = torso_ratio
        self.center_ratio = center_ratio
        self.outlier_factor = outlier_factor
        self.v_weight = v_weight
        self.dark_v_floor = dark_v_floor
        self.team_kmeans: KMeans | None = None
        self.team_colors: dict[int, np.ndarray] = {}   # 0/1 -> màu BGR trung bình cụm (để hiển thị/JSON)
        self._outlier_thresh: float = float("inf")     # ngưỡng ngoại lai (không gian đặc trưng)
        self._track_votes: dict[int, Counter] = defaultdict(Counter)  # track_id -> phiếu đội
        self._warmup_colors: list[np.ndarray] = []     # màu BGR gom trong cửa sổ khởi động

    # ---------- BGR -> đặc trưng HSV (bền với ánh sáng) ----------
    # Áo trắng: S thấp. Áo màu (xanh-lá/đỏ/xanh dương): S cao + H đặc trưng.
    # Hue vòng tròn -> mã hoá cos/sin (nhân S vì hue của pixel ít bão hoà là vô nghĩa).
    # V nhân v_weight (<1) để bóng đổ ít làm lệch phân cụm.
    def _feat(self, bgr) -> np.ndarray:
        if self.feature_space == "bgr":
            return np.asarray(bgr, np.float32)
        px = np.uint8([[[int(bgr[0]), int(bgr[1]), int(bgr[2])]]])
        h, s, v = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
        ang = h / 180.0 * 2 * np.pi
        return np.array([np.cos(ang) * s, np.sin(ang) * s, s, v * self.v_weight], np.float32)

    # ---------- lấy màu áo của 1 bbox ----------
    def get_player_color(self, frame: np.ndarray, bbox) -> np.ndarray | None:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None
        crop = frame[y1:y2, x1:x2]
        # DẢI THÂN dùng lấy màu áo. Mặc định 0..torso_ratio (nửa trên).
        # Đặt torso_top>0 để BỎ ĐẦU (da/tóc) — nguồn lẫn tạp chính.
        h_c = crop.shape[0]
        t0 = int(h_c * self.torso_top)
        t1 = int(h_c * (self.torso_bot if self.torso_bot is not None else self.torso_ratio))
        t1 = max(t0 + 1, min(h_c, t1))
        torso = crop[t0:t1, :]
        # giữ DẢI GIỮA theo chiều ngang (bỏ mép trái/phải -> tránh dính người kế bên)
        cw = torso.shape[1]
        side = int(cw * (1 - self.center_ratio) / 2)
        torso = torso[:, side: cw - side] if cw - 2 * side >= 2 else torso
        pixels = torso.reshape(-1, 3).astype(np.float32)
        if len(pixels) < 4:
            return None
        # crop gần như 1 màu (áo lấp đầy khung) -> khỏi tách nền, trả màu trung bình
        if len(np.unique(pixels, axis=0)) < 2:
            return pixels.mean(axis=0)
        # KMeans 2 cụm tách áo khỏi cỏ. CỎ nằm ở 4 GÓC crop (cầu thủ ở giữa, bbox
        # có viền cỏ) -> cụm chiếm góc = cỏ, cụm còn lại = áo. Đây là cách ĐÃ KIỂM
        # CHỨNG chạy đúng trên video broadcast thật (kể cả đội áo xanh-lá trên sân cỏ).
        km = KMeans(n_clusters=2, n_init=3, random_state=0).fit(pixels)
        labels = km.labels_.reshape(torso.shape[0], torso.shape[1])
        corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        bg_label = Counter(corners).most_common(1)[0][0]
        jersey_label = 1 - bg_label
        return km.cluster_centers_[jersey_label]  # màu BGR

    # ---------- gom màu trong cửa sổ khởi động ----------
    # QUAN TRỌNG: mỗi màu phải lấy từ ĐÚNG frame chứa bbox đó.
    def collect(self, frame: np.ndarray, player_bboxes) -> None:
        for b in player_bboxes:
            c = self.get_player_color(frame, b)
            if c is not None:
                self._warmup_colors.append(c)

    def fit_from_buffer(self, min_samples: int = 6) -> bool:
        if len(self._warmup_colors) < max(2, min_samples):
            return False
        bgr = np.array(self._warmup_colors)
        feats = np.array([self._feat(c) for c in bgr])
        self.team_kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(feats)
        labels = self.team_kmeans.labels_
        for t in (0, 1):
            members = bgr[labels == t]
            # màu hiển thị = trung bình BGR các thành viên cụm (để vẽ/JSON cho dễ đọc)
            self.team_colors[t] = members.mean(axis=0) if len(members) else bgr.mean(axis=0)
        # ngưỡng ngoại lai tính trong KHÔNG GIAN ĐẶC TRƯNG (nơi thực sự phân cụm)
        c0, c1 = self.team_kmeans.cluster_centers_
        d = float(np.linalg.norm(c0 - c1))
        self._outlier_thresh = self.outlier_factor * (d / 2)
        return True

    # tiện dụng: fit ngay trên 1 frame
    def fit(self, frame: np.ndarray, player_bboxes) -> bool:
        self.collect(frame, player_bboxes)
        return self.fit_from_buffer(min_samples=2)

    # ---------- phân loại MỘT detection, KHÔNG tích luỹ (dùng cho bầu cửa sổ trượt) ----------
    def classify(self, frame: np.ndarray, bbox):
        """Trả (team|None, weight). team=None nghĩa là màu ngoại lai (áo lạ/không tin được)."""
        if self.team_kmeans is None:
            return None, 0.0
        color = self.get_player_color(frame, bbox)
        if color is None:
            return None, 0.0
        if self.feature_space == "hsv" and float(max(color)) < self.dark_v_floor:
            return None, 0.0
        feat = self._feat(color).reshape(1, -1)
        team = int(self.team_kmeans.predict(feat)[0])
        dist = float(np.linalg.norm(feat[0] - self.team_kmeans.cluster_centers_[team]))
        if dist > self._outlier_thresh:
            return None, 0.0
        x1, y1, x2, y2 = map(float, bbox)
        return team, max(1.0, (x2 - x1) * (y2 - y1))

    # ---------- gán đội cho 1 track (bỏ phiếu CÓ TRỌNG SỐ + lọc ngoại lai) ----------
    def assign(self, frame: np.ndarray, bbox, track_id: int) -> int | None:
        if self.team_kmeans is None:
            return None
        self._seen_frames[track_id] += 1
        color = self.get_player_color(frame, bbox)
        if color is not None:
            v_raw = float(max(color))  # V trong HSV = kênh sáng nhất của BGR
            if self.feature_space == "hsv" and v_raw < self.dark_v_floor:
                # áo quá tối (GK áo đen) -> ngoại lai, không bỏ phiếu đội
                return self._track_votes[track_id].most_common(1)[0][0] if self._track_votes[track_id] else None
            feat = self._feat(color).reshape(1, -1)
            team = int(self.team_kmeans.predict(feat)[0])
            # khoảng cách (đặc trưng) tới tâm cụm: quá xa -> ngoại lai (GK áo lạ), bỏ phiếu
            dist = float(np.linalg.norm(feat[0] - self.team_kmeans.cluster_centers_[team]))
            if dist <= self._outlier_thresh:
                # TRỌNG SỐ = diện tích box: box to/gần (áo rõ) phiếu nặng, box nhỏ/xa
                # (dễ dính cỏ) phiếu nhẹ -> vài frame nhiễu không lật cả track.
                x1, y1, x2, y2 = map(float, bbox)
                w = max(1.0, (x2 - x1) * (y2 - y1))
                self._track_votes[track_id][team] += w
                self._vote_frames[track_id] += 1
        if not self._track_votes[track_id]:
            return None
        return self._track_votes[track_id].most_common(1)[0][0]

    # tỉ lệ frame bỏ được phiếu hợp lệ / frame xuất hiện. Thấp = màu không đáng tin
    # (áo lạ như GK) -> không nên nhận đội từ vài phiếu nhiễu.
    def vote_coverage(self, track_id: int) -> float:
        seen = self._seen_frames.get(track_id, 0)
        return (self._vote_frames.get(track_id, 0) / seen) if seen else 0.0

    # đọc đội chốt (majority). min_coverage: dưới ngưỡng này coi như KHÔNG có đội.
    def final_team(self, track_id: int, min_coverage: float = 0.0) -> int | None:
        v = self._track_votes.get(track_id)
        if not v:
            return None
        if self.vote_coverage(track_id) < min_coverage:
            return None
        return v.most_common(1)[0][0]