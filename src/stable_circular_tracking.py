"""
Stable circular clamp tracking.

Pipeline:
1) YOLO11 detector
2) BoT-SORT temporal tracking
3) Circular motion model
4) Hungarian one-to-one assignment
5) Angular gating + motion prediction during temporary dropouts

The displayed Error (%) is a frame-level detection-count error, not a MOT metric.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


MAX_CLAMPS = 14
ANGLE_GATE_DEG = 25.0


def normalize_angle(a: float) -> float:
    return (a + 2 * math.pi) % (2 * math.pi)


def angle_difference(a: float, b: float) -> float:
    d = abs(a - b)
    return min(d, 2 * math.pi - d)


def fit_circle(points):
    """Least-squares circle-center fit for [(x, y), ...]."""
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)

    A = np.column_stack((2 * x, 2 * y, np.ones(len(points))))
    b = x * x + y * y

    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return float(c[0]), float(c[1])


def run(weights: str, input_video: str, tracker_yaml: str, output_video: str,
        conf: float = 0.35, iou: float = 0.50, imgsz: int = 960) -> None:
    model = YOLO(weights)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    initialized = False
    circle_cx = circle_cy = None

    # Logical Clamp ID -> predicted angular state
    track_angles = {}
    angular_velocity = {}
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        results = model.track(
            frame,
            persist=True,
            tracker=tracker_yaml,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )
        r = results[0]

        detections = []
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for box, score in zip(boxes, confs):
                x1, y1, x2, y2 = box
                detections.append(
                    {
                        "box": box,
                        "cx": (x1 + x2) / 2,
                        "cy": (y1 + y2) / 2,
                        "conf": float(score),
                    }
                )

        detected_count = len(detections)

        # Initialize when all 14 clamps are visible.
        if not initialized and detected_count == MAX_CLAMPS:
            centers = [(d["cx"], d["cy"]) for d in detections]
            circle_cx, circle_cy = fit_circle(centers)

            for d in detections:
                d["angle"] = normalize_angle(
                    math.atan2(d["cy"] - circle_cy, d["cx"] - circle_cx)
                )

            detections.sort(key=lambda d: d["angle"])

            for i, d in enumerate(detections):
                logical_id = i + 1
                track_angles[logical_id] = d["angle"]
                angular_velocity[logical_id] = 0.0
                d["logical_id"] = logical_id

            initialized = True
            print(f"Tracker initialized at frame {frame_number}")

        elif initialized:
            predicted = {
                lid: normalize_angle(track_angles[lid] + angular_velocity[lid])
                for lid in range(1, MAX_CLAMPS + 1)
            }

            for d in detections:
                d["angle"] = normalize_angle(
                    math.atan2(d["cy"] - circle_cy, d["cx"] - circle_cx)
                )

            if detected_count > 0:
                logical_ids = list(range(1, MAX_CLAMPS + 1))
                cost = np.zeros((MAX_CLAMPS, detected_count), dtype=float)

                for i, lid in enumerate(logical_ids):
                    for j, d in enumerate(detections):
                        cost[i, j] = angle_difference(predicted[lid], d["angle"])

                rows, cols = linear_sum_assignment(cost)

                for row, col in zip(rows, cols):
                    lid = logical_ids[row]

                    if cost[row, col] > math.radians(ANGLE_GATE_DEG):
                        continue

                    d = detections[col]
                    d["logical_id"] = lid

                    old = track_angles[lid]
                    new = d["angle"]

                    delta = new - old
                    if delta > math.pi:
                        delta -= 2 * math.pi
                    elif delta < -math.pi:
                        delta += 2 * math.pi

                    angular_velocity[lid] = (
                        0.85 * angular_velocity[lid] + 0.15 * delta
                    )
                    track_angles[lid] = new

            # Predict missing clamps through short detection dropouts.
            assigned = {
                d["logical_id"] for d in detections if "logical_id" in d
            }
            for lid in range(1, MAX_CLAMPS + 1):
                if lid not in assigned:
                    track_angles[lid] = normalize_angle(
                        track_angles[lid] + angular_velocity[lid]
                    )

        # Draw assigned detections.
        for d in detections:
            if "logical_id" not in d:
                continue

            x1, y1, x2, y2 = map(int, d["box"])
            lid = d["logical_id"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Clamp {lid} {d['conf']:.2f}",
                (x1, max(y1 - 6, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        count_error = abs(MAX_CLAMPS - detected_count) / MAX_CLAMPS * 100

        cv2.rectangle(frame, (20, 20), (330, 110), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"Detected: {detected_count} / {MAX_CLAMPS}",
            (35, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Error: {count_error:.1f}%",
            (35, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if initialized and detected_count < 10:
            cv2.putText(
                frame,
                "DETECTION DROP",
                (width - 400, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved: {output_video}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to trained best.pt")
    parser.add_argument("--input", required=True, help="Input video")
    parser.add_argument(
        "--tracker",
        default="configs/custom_botsort_v3.yaml",
        help="BoT-SORT YAML config",
    )
    parser.add_argument(
        "--output",
        default="results/V3_STABLE_CIRCULAR_IDS.mp4",
        help="Output video",
    )
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=960)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        weights=args.weights,
        input_video=args.input,
        tracker_yaml=args.tracker,
        output_video=args.output,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
