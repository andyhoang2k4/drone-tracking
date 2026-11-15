"""Ve ket qua pipeline (muc tieu + state Motion Memory) va moi track BoT-SORT len video.

Chi doc log (khong chay model), dung de xem bang mat va chon --target-track-id::

    # moi track + ID (de chon muc tieu), tu log cua `python -m src.track`
    python scripts/render_overlay.py --video V.mp4 --tracks tracks.jsonl --out overlay.mp4
    # them muc tieu + state tu log cua scripts/run_pipeline.py
    python scripts/render_overlay.py --video V.mp4 --tracks tracks.jsonl \
        --pipeline experiments/<id>/logs/pipeline.jsonl --out overlay.mp4

Mau khung muc tieu: TRACKING xanh la, LOST vang cam, REIDENTIFYING xanh duong, FAILED do.
Track khac: khung xam mong + ID. Video ra thu nho ve --width (mac dinh 1280).
--video phai la video (khong ho tro thu muc anh); frame_idx cua log tinh tu 0 theo thu tu frame doc duoc.
"""
import argparse, json
import cv2

COLORS = {"TRACKING": (0, 200, 0), "LOST": (0, 200, 255), "REIDENTIFYING": (255, 120, 0), "FAILED": (0, 0, 255)}

def load(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["frame_idx"]: r for r in map(json.loads, f)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--pipeline")
    ap.add_argument("--tracks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1280)
    a = ap.parse_args()
    if not a.pipeline and not a.tracks:
        ap.error("can it nhat mot trong --tracks / --pipeline")
    pipe, trk = load(a.pipeline), load(a.tracks)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        ap.error(f"khong mo duoc video: {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = a.width / w0
    size = (a.width, int(h0 * s))
    out = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, size)
        for t in trk.get(i, {}).get("tracks", []):
            x1, y1, x2, y2 = [int(v * s) for v in t["bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (160, 160, 160), 1)
            cv2.putText(img, str(t["track_id"]), (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        p = pipe.get(i)
        if p:
            st = p["state"]
            box = p.get("predicted_bbox")
            if box:
                x1, y1, x2, y2 = [int(v * s) for v in box]
                cv2.rectangle(img, (x1, y1), (x2, y2), COLORS.get(st, (255, 255, 255)), 2)
            txt = f"f{i} target id={p.get('track_id')} {st} motion_conf={p.get('motion_confidence') or 0:.2f} since={p.get('frames_since_confirmed')}"
            cv2.rectangle(img, (0, 0), (size[0], 26), (0, 0, 0), -1)
            cv2.putText(img, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS.get(st, (255, 255, 255)), 1)
        out.write(img)
        i += 1
    out.release()
    print(f"Da ghi {a.out}: {i} frame, {size[0]}x{size[1]}")

if __name__ == "__main__":
    main()
