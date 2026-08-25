import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--fmt", default="yolov11")
    ap.add_argument("--out", default="datasets")
    a = ap.parse_args()

    from roboflow import Roboflow

    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        from secrets_config import ROBOFLOW_API_KEY as key
    assert key, "Thiếu ROBOFLOW_API_KEY (env hoặc secrets_config.py)"

    rf = Roboflow(api_key=key)
    proj = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    ds = proj.version(a.version).download(
        a.fmt, location=os.path.join(a.out, "football-players-detection-1"))
    print("Đã tải về:", ds.location)
    print("data.yaml:", os.path.join(ds.location, "data.yaml"))


if __name__ == "__main__":
    main()