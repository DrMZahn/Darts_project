import os
from pathlib import Path

# Pfade
JSON_PATH = "dartboard_points_calibration.json"
YOLO_WEIGHTS = "/home/matthias/PRIV/Darts_ML/runs/detect/runs/darts_gpu_licht_beide_seiten/train/weights/best.pt"
YOLO_MODEL_BOARD_PATH = "/home/matthias/PRIV/Darts_ML_detect_board/runs/detect/runs/darts_gpu_board/train3/weights/best.pt"
CAM_INDEX = 2

# Dartboard-Geometrie (FULL HD)
TARGET_SIZE = 2000
CENTER = TARGET_SIZE // 2
RADIUS = 480
R_DOUBLE_OUTER = RADIUS
R_DOUBLE_INNER = int(RADIUS * (160 / 170))
R_TRIPLE_OUTER = int(RADIUS * 110 / 170)
R_TRIPLE_INNER = int(RADIUS * (100 / 170))
R_OUTER_BULL = int(RADIUS * ((31.8 / 2) / 170))
R_INNER_BULL = int(RADIUS * ((14 / 2) / 170))

# Kamera
ORIG_W, ORIG_H = 1920, 1080
NEW_W, NEW_H = 1920, 1080
scale_x = NEW_W / ORIG_W
scale_y = NEW_H / ORIG_H

# Segmente (Standard-Dartscheibe)
SEGMENTS = [
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
    3, 19, 7, 16, 8, 11, 14, 9, 12, 5
]

# Kalibrierungspunkte
POINT_NAMES = ["Mitte", "P_05_20", "P_13_06", "P_17_03", "P_08_11"]

# Auto-Detection
DETECTION_COOLDOWN = 2.0
