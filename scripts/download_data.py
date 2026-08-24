import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--fmt", default="yolov11")
    ap.add_argument("--out", default="datasets")
    a = ap.parse_args()

    from roboflow import Roboflow

    key = os.environ.get("ROBOFLOW_API_KEY")
    assert key, "Thiếu env ROBOFLOW_API_KEY"
    rf = Roboflow(api_key=key)
    proj = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    ds = proj.version(a.version).download(a.fmt, location=os.path.join(a.out, "football-players-detection-1"))
    print("Đã tải về:", ds.location)
    print("data.yaml:", os.path.join(ds.location, "data.yaml"))


if __name__ == "__main__":
    main()
