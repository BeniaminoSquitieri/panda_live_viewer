"""One-shot wrist-camera diagnostic.

Grabs a single frame from the wrist RGB (compressed) topic, saves it to a PNG so
you can *see* what the camera sees, then runs the OWLv2 zero-shot detector at a
very low threshold and prints EVERY candidate box + score. This tells you
whether the object is visible/detected at all (even below the live threshold) or
not seen at all (framing/lighting problem).

Run from ~/panda_live_viewer (with the CycloneDDS env exported):

    python3 -m scripts.wrist_snapshot \
        --topic /panda/camera/wrist/image_compressed \
        --registry $HOME/panda_live_viewer/perception/registries/coffee_registry.json \
        --model google/owlv2-base-patch16-ensemble \
        --threshold 0.02 \
        --out /tmp/wrist_snapshot.png
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
from sensor_msgs.msg import CompressedImage


def _grab_frame(topic: str, timeout_s: float) -> np.ndarray:
    rclpy.init()
    node = Node("wrist_snapshot")
    holder: dict[str, np.ndarray] = {}

    def _cb(msg: CompressedImage) -> None:
        if "img" in holder:
            return
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
        if img is not None:
            holder["img"] = img

    node.create_subscription(CompressedImage, topic, _cb, 10)
    deadline = time.time() + timeout_s
    while rclpy.ok() and "img" not in holder and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    if "img" not in holder:
        raise SystemExit(f"No frame received on {topic} within {timeout_s}s.")
    return holder["img"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/panda/camera/wrist/image_compressed")
    parser.add_argument("--registry", default="")
    parser.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--out", default="/tmp/wrist_snapshot.png")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    bgr = _grab_frame(args.topic, args.timeout)
    print(f"[snapshot] frame shape={bgr.shape} dtype={bgr.dtype}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, bgr)
    print(f"[snapshot] saved RGB frame -> {args.out}")
    print("[snapshot] open it with:  code " + args.out + "   (or scp it to your laptop)")

    # Build candidate labels from the registry (natural-language aliases).
    labels: list[str]
    if args.registry:
        registry = json.loads(Path(args.registry).read_text())
        from perception.segmenters import labels_from_registry

        labels = labels_from_registry(registry)
    else:
        labels = ["coffee capsule", "espresso capsule", "cup", "coffee machine", "plate"]
    print(f"[snapshot] candidate labels: {labels}")

    # Run OWLv2 directly (transformers zero-shot pipeline) and print all scores.
    from PIL import Image
    from transformers import pipeline

    rgb = bgr[:, :, ::-1]
    pil = Image.fromarray(np.ascontiguousarray(rgb))
    detector = pipeline(model=args.model, task="zero-shot-object-detection")
    results = detector(pil, candidate_labels=labels, threshold=args.threshold)
    results = sorted(results, key=lambda r: r["score"], reverse=True)
    print(f"[snapshot] {len(results)} detections at threshold {args.threshold}:")
    if not results:
        print("  (NOTHING detected even at this very low threshold -> object likely not")
        print("   visible / too small / bad lighting in the wrist view.)")
    for r in results:
        box = r["box"]
        print(
            f"  {r['label']:<22} score={r['score']:.4f} "
            f"box=({box['xmin']},{box['ymin']})-({box['xmax']},{box['ymax']})"
        )


if __name__ == "__main__":
    main()
