import argparse, glob, os, cv2, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--frames-dir", default="frames_smoke", help="thu muc anh PNG (tuong doi voi thu muc dang chay)")
args = ap.parse_args()
fs = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")), key=os.path.getmtime)
print("so file:", len(fs))
im = cv2.imread(fs[-1]); print(os.path.basename(fs[-1]), im.shape)
b, g, r = [im[..., i].astype(float) for i in range(3)]
for name, c, o1, o2 in (("do", r, g, b), ("xanh la", g, r, b), ("xanh duong", b, r, g)):
    m = (c > 80) & (c > 1.8 * o1) & (c > 1.8 * o2)
    ys, xs = np.nonzero(m)
    box = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())) if len(xs) else None
    print(name, "pixel:", int(m.sum()), "bbox(x0,x1,y0,y1):", box)
