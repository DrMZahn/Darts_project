import cv2
import numpy as np
import math
import time
import os

DART_NUMBERS = [
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
    3, 19, 7, 16, 8, 11, 14, 9, 12, 5
]

def calculate_score(center, hit_point, radii):
    dx = hit_point[0] - center[0]
    dy = center[1] - hit_point[1]
    distance = math.hypot(dx, dy)

    angle = math.degrees(math.atan2(dx, dy))
    if angle < 0:
        angle += 360

    sector = int((angle + 9) // 18) % 20
    base_score = DART_NUMBERS[sector]

    r_bull, r_outer_bull, r_triple, r_double = radii

    if distance < r_bull:
        return 50
    elif distance < r_outer_bull:
        return 25
    elif r_triple[0] < distance < r_triple[1]:
        return base_score * 3
    elif r_double[0] < distance < r_double[1]:
        return base_score * 2
    elif distance < r_double[1]:
        return base_score
    else:
        return 0


def detect_dart_score_with_debug(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Bild konnte nicht geladen werden")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)

    # --- Dartboard erkennen ---
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=300,
        param1=100,
        param2=30,
        minRadius=200,
        maxRadius=400
    )

    if circles is None:
        raise RuntimeError("Keine Dartsscheibe erkannt")

    cx, cy, r = map(int, circles[0][0])

    radii = (
        r * 0.04,
        r * 0.09,
        (r * 0.53, r * 0.58),
        (r * 0.93, r * 1.0)
    )

    # --- Dart erkennen ---
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=60,
        maxLineGap=10
    )

    if lines is None:
        raise RuntimeError("Kein Dartpfeil erkannt")

    longest = max(lines, key=lambda l: math.hypot(
        l[0][0] - l[0][2], l[0][1] - l[0][3]
    ))[0]

    x1, y1, x2, y2 = longest

    d1 = math.hypot(x1 - cx, y1 - cy)
    d2 = math.hypot(x2 - cx, y2 - cy)
    tip = (x1, y1) if d1 < d2 else (x2, y2)

    score = calculate_score((cx, cy), tip, radii)

    # --- Debug-Zeichnung ---
    debug = img.copy()

    # Dartboard
    cv2.circle(debug, (cx, cy), r, (0, 255, 0), 2)
    cv2.circle(debug, (cx, cy), 5, (0, 0, 255), -1)

    # Dart
    cv2.line(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.circle(debug, tip, 7, (0, 0, 255), -1)

    # Score Text
    cv2.putText(
        debug,
        f"Score: {score}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    # --- Speichern ---
    os.makedirs("images/test", exist_ok=True)
    timestamp = int(time.time())
    out_path = f"images/test/darts_debug_{timestamp}.jpg"
    cv2.imwrite(out_path, debug)

    return {
        "score": score,
        "dart_tip": tip,
        "center": (cx, cy),
        "debug_image": out_path
    }
