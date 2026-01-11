"""Detect and count cups from a webcam feed using OpenCV.

Usage:
  python3 cup_counter.py --camera 0

Press 'q' to quit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    contour: np.ndarray
    center: tuple[int, int]
    area: float
    aspect_ratio: float
    solidity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count cups using a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument(
        "--min-area",
        type=int,
        default=2000,
        help="Minimum contour area to consider as a cup.",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=50000,
        help="Maximum contour area to consider as a cup.",
    )
    parser.add_argument(
        "--aspect-min",
        type=float,
        default=0.45,
        help="Minimum aspect ratio (w/h) for a cup contour.",
    )
    parser.add_argument(
        "--aspect-max",
        type=float,
        default=1.2,
        help="Maximum aspect ratio (w/h) for a cup contour.",
    )
    parser.add_argument(
        "--solidity-min",
        type=float,
        default=0.6,
        help="Minimum solidity for a cup contour.",
    )
    return parser.parse_args()


def preprocess_frame(frame: np.ndarray, subtractor: cv2.BackgroundSubtractor) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    fg_mask = subtractor.apply(blurred)
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed


def contour_solidity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return float(area / hull_area)


def detect_cups(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
    aspect_min: float,
    aspect_max: float,
    solidity_min: float,
) -> list[Detection]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue
        aspect_ratio = w / float(h)
        if not (aspect_min <= aspect_ratio <= aspect_max):
            continue
        solidity = contour_solidity(contour)
        if solidity < solidity_min:
            continue
        center = (int(x + w / 2), int(y + h / 2))
        detections.append(
            Detection(
                contour=contour,
                center=center,
                area=area,
                aspect_ratio=aspect_ratio,
                solidity=solidity,
            )
        )
    return detections


def draw_detections(frame: np.ndarray, detections: Iterable[Detection]) -> None:
    for detection in detections:
        x, y, w, h = cv2.boundingRect(detection.contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.circle(frame, detection.center, 4, (0, 0, 255), -1)


def main() -> None:
    args = parse_args()
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit("No se pudo abrir la cámara.")

    subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50)

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        mask = preprocess_frame(frame, subtractor)
        detections = detect_cups(
            mask,
            min_area=args.min_area,
            max_area=args.max_area,
            aspect_min=args.aspect_min,
            aspect_max=args.aspect_max,
            solidity_min=args.solidity_min,
        )
        draw_detections(frame, detections)
        cv2.putText(
            frame,
            f"Vasos detectados: {len(detections)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Contador de vasos", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
