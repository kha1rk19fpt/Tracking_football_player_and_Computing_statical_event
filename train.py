import argparse

PROFILES = {
    "local4060": dict(model="yolo26s.pt", batch=16, device=0),
    "t4":        dict(model="yolo26m.pt", batch=16, device=0),
    "cpu":       dict(model="yolo26n.pt", batch=4,  device="cpu"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="đường dẫn data.yaml")
    ap.add_argument("--profile", default="local4060", choices=PROFILES)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)  # giai đoạn 1 = 640
    ap.add_argument("--model", default=None, help="ghi đè model của preset")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--name", default="phase1")
    a = ap.parse_args()

    from ultralytics import YOLO

    p = PROFILES[a.profile]
    model_w = a.model or p["model"]
    batch = a.batch or p["batch"]

    model = YOLO(model_w)
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=batch,
        device=p["device"],
        name=a.name,
        patience=20,          # early stop nếu 20 epoch không cải thiện
        cos_lr=True,
        # augment nhẹ, hợp bối cảnh sân cỏ (không lật dọc, không xoay mạnh)
        fliplr=0.5, flipud=0.0, degrees=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        mosaic=1.0, close_mosaic=10,  # tắt mosaic 10 epoch cuối cho ổn định box
        plots=True,
    )
    # đánh giá lại trên val, in mAP để chốt acceptance
    metrics = model.val(data=a.data, imgsz=a.imgsz)
    print("mAP50-95:", round(float(metrics.box.map), 4),
          "| mAP50:", round(float(metrics.box.map50), 4))
    print("Per-class mAP50:", {model.names[i]: round(float(v), 3)
                               for i, v in enumerate(metrics.box.maps)})


if __name__ == "__main__":
    main()
