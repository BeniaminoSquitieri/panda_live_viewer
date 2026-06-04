"""One-shot camera diagnostic: RGB + depth + OWLv2 scores.

Grabs a single frame from a camera's RGB (compressed) topic AND its depth topic,
saves both to a *visible* folder (default ~/panda_live_viewer/snapshots) so you
can see exactly what the camera sees, then runs the OWLv2 zero-shot detector at
a very low threshold and prints every candidate box + score, plus the depth
statistics inside each detection box (to diagnose implausible far poses).

Run from ~/panda_live_viewer (with the CycloneDDS env exported):

    python3 -m scripts.wrist_snapshot \
        --topic /panda/camera/front/image_compressed \
        --depth-topic /panda/camera/front/depth \
        --registry $HOME/panda_live_viewer/perception/registries/coffee_registry.json \
        --model google/owlv2-base-patch16-ensemble \
        --threshold 0.02
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image


def _grab(topic, msg_type, decode, timeout_s):
    rclpy.init()
    node = Node("camera_snapshot")
    holder = {}

    def _cb(msg) -> None:
        if "data" in holder:
            return
        decoded = decode(msg)
        if decoded is not None:
            holder["data"] = decoded

    node.create_subscription(msg_type, topic, _cb, 10)
    deadline = time.time() + timeout_s
    while rclpy.ok() and "data" not in holder and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    return holder.get("data")


def _decode_rgb(msg: CompressedImage):
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR


def _decode_depth(msg: Image):
    encoding = str(getattr(msg, "encoding", "")).upper()
    width, height = int(msg.width), int(msg.height)
    if encoding in {"16UC1", "MONO16"}:
        dtype = np.uint16
    elif encoding == "32FC1":
        dtype = np.float32
    else:
        print(f"[snapshot] unsupported depth encoding {encoding!r}")
        return None
    arr = np.frombuffer(bytes(msg.data), dtype=dtype)
    expected = width * height
    if arr.size < expected:
        return None
    return arr[:expected].reshape((height, width))


def _depth_to_meters(depth: np.ndarray) -> np.ndarray:
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) * 0.001
    return depth.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/panda/camera/front/image_compressed")
    parser.add_argument("--depth-topic", default="/panda/camera/front/depth")
    parser.add_argument("--registry", default="")
    parser.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument(
        "--out-dir",
        default=str(Path.home() / "panda_live_viewer" / "snapshots"),
        help="Visible folder to save RGB + depth PNGs.",
    )
    parser.add_argument("--name", default="snapshot")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = _grab(args.topic, CompressedImage, _decode_rgb, args.timeout)
    if bgr is None:
        raise SystemExit(f"No RGB frame on {args.topic} within {args.timeout}s.")
    rgb_path = out_dir / f"{args.name}_rgb.png"
    cv2.imwrite(str(rgb_path), bgr)
    print(f"[snapshot] RGB  shape={bgr.shape} -> {rgb_path}")

    depth = _grab(args.depth_topic, Image, _decode_depth, args.timeout)
    depth_m = None
    if depth is None:
        print(f"[snapshot] WARNING: no depth frame on {args.depth_topic}")
    else:
        depth_m = _depth_to_meters(depth)
        finite = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
        if finite.size:
            print(
                f"[snapshot] DEPTH shape={depth.shape} dtype={depth.dtype} "
                f"min={finite.min():.3f}m median={np.median(finite):.3f}m "
                f"max={finite.max():.3f}m  (valid {finite.size}/{depth_m.size})"
            )
        vis = depth_m.copy()
        vis[~np.isfinite(vis)] = 0
        vmax = np.percentile(vis[vis > 0], 95) if np.any(vis > 0) else 1.0
        norm = np.clip(vis / max(vmax, 1e-6), 0, 1)
        color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        color[vis <= 0] = 0
        depth_path = out_dir / f"{args.name}_depth.png"
        cv2.imwrite(str(depth_path), color)
        print(f"[snapshot] DEPTH colorized -> {depth_path}")
        if depth_m.shape != bgr.shape[:2]:
            print(
                f"[snapshot] WARNING: depth {depth_m.shape} != rgb {bgr.shape[:2]} "
                "(RGB/depth NOT aligned -> poses will be wrong)"
            )

    if args.registry:
        registry = json.loads(Path(args.registry).read_text())
        from perception.segmenters import labels_from_registry

        labels = labels_from_registry(registry)
    else:
        labels = ["coffee capsule", "espresso capsule", "cup", "coffee machine", "plate"]
    print(f"[snapshot] candidate labels: {labels}")

    from PIL import Image as PILImage
    from transformers import pipeline

    rgb = bgr[:, :, ::-1]
    pil = PILImage.fromarray(np.ascontiguousarray(rgb))
    detector = pipeline(model=args.model, task="zero-shot-object-detection")
    results = detector(pil, candidate_labels=labels, threshold=args.threshold)
    results = sorted(results, key=lambda r: r["score"], reverse=True)
    print(f"[snapshot] {len(results)} detections at threshold {args.threshold}:")
    if not results:
        print("  (NOTHING detected even at this very low threshold.)")
    for r in results:
        b = r["box"]
        line = (
            f"  {r['label']:<22} score={r['score']:.4f} "
            f"box=({b['xmin']},{b['ymin']})-({b['xmax']},{b['ymax']})"
        )
        if depth_m is not None and depth_m.shape == bgr.shape[:2]:
            roi = depth_m[b["ymin"]:b["ymax"], b["xmin"]:b["xmax"]]
            roi = roi[np.isfinite(roi) & (roi > 0)]
            if roi.size:
                line += f"  depth[min/med/max]={roi.min():.2f}/{np.median(roi):.2f}/{roi.max():.2f}m"
            else:
                line += "  depth=NONE-in-box"
        print(line)


if __name__ == "__main__":
    main()
