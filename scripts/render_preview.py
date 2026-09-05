"""Render an STL mesh to a multi-angle preview PNG (used for the README image).

Usage:
    python render_preview.py input.STL output.png

Dependencies: numpy, matplotlib
"""
import struct
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_stl(path):
    with open(path, "rb") as f:
        head = f.read(512)
    if b"solid" in head[:64] and b"facet" in head:
        tris = []
        with open(path, "r", errors="ignore") as f:
            cur = []
            for line in f:
                p = line.split()
                if p and p[0] == "vertex":
                    cur.append([float(p[1]), float(p[2]), float(p[3])])
                    if len(cur) == 3:
                        tris.append(cur)
                        cur = []
        tris = np.array(tris, dtype=np.float32)
        v1 = tris[:, 1] - tris[:, 0]
        v2 = tris[:, 2] - tris[:, 0]
        normals = np.cross(v1, v2)
    else:
        with open(path, "rb") as f:
            f.read(80)
            n = struct.unpack("<I", f.read(4))[0]
            data = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
        floats = data[:, :48].copy().view("<f4").reshape(n, 4, 3)
        normals, tris = floats[:, 0], floats[:, 1:4].astype(np.float32)
    return normals, tris


def main():
    stl_path = sys.argv[1] if len(sys.argv) > 1 else "model.STL"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "preview.png"

    normals, tris = load_stl(stl_path)
    print("triangles:", len(tris))

    all_pts = tris.reshape(-1, 3)
    mn, mx = all_pts.min(0), all_pts.max(0)
    center, span = (mn + mx) / 2, (mx - mn).max()
    print("bbox min:", mn.round(2), "max:", mx.round(2))

    views = [(32, -55), (30, 55), (16, -135)]
    fig = plt.figure(figsize=(15, 5.8), facecolor="#f7f7f9")
    light_dir = np.array([0.4, -0.7, 0.6])
    light_dir = light_dir / np.linalg.norm(light_dir)

    for i, (elev, azim) in enumerate(views, 1):
        ax = fig.add_subplot(1, 3, i, projection="3d", facecolor="#f7f7f9")
        n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
        shade = 0.35 + 0.65 * np.clip(n @ light_dir, 0, 1)
        base = np.array([0.28, 0.55, 0.85])
        colors = np.clip(shade[:, None] * base, 0, 1)
        pc = Poly3DCollection(tris, facecolors=colors, edgecolors="none")
        ax.add_collection3d(pc)
        r = span / 2 * 1.12
        ax.set_xlim(center[0] - r, center[0] + r)
        ax.set_ylim(center[1] - r, center[1] + r)
        ax.set_zlim(center[2] - r, center[2] + r)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_proj_type("ortho")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", pad_inches=0.05)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
