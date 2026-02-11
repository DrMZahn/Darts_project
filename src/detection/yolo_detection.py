import cv2
import numpy as np
import time
from ultralytics import YOLO
from typing import Tuple, Optional, List
from src.config.constants import YOLO_WEIGHTS, JSON_PATH
from src.scoring.scoring_logic import compute_score_from_tip

def get_best_dart_tip(results) -> Tuple[Optional[Tuple[int, int]], int, int]:
    """Extrahiert beste Dartspitze aus YOLO-Ergebnissen"""
    best_tip = None
    center_x, center_y = 0, 0
    
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                best_tip = (center_x, center_y)
                print(f"🎯 Dart-Tip gefunden: ({center_x}, {center_y})")
                return best_tip, center_x, center_y
    
    print("❌ Kein Dart erkannt")
    return None, 0, 0

def get_score_from_tip(best_tip: Optional[Tuple[int, int]], H: np.ndarray) -> Tuple[int, str]:
    """Berechnet Score aus Dartspitze mit Homographie"""
    if best_tip is None:
        return 0, " 0"
    
    try:
        point_array = np.array([[[float(best_tip[0]), float(best_tip[1])]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(point_array, H)
        exact_tip = (int(warped[0, 0, 0]), int(warped[0, 0, 1]))
        
        points, str_points = compute_score_from_tip(exact_tip)
        print(f"📍 Tip: {best_tip} → {exact_tip} → {points} {str_points}")
        return points, str_points
        
    except Exception as e:
        print(f"❌ Transform Fehler: {e}")
        return 0, "ERR"

def analyze_single_dart(webcam, refimg: np.ndarray, H: np.ndarray) -> Tuple[int, str, np.ndarray, np.ndarray]:
    """Komplettanalyse eines einzelnen Darts"""
    img = webcam.capture_frame()
    h, w = img.shape[:2]
    refimg_resized = cv2.resize(refimg, (w, h))
    
    # Diff-Berechnung
    diff = cv2.absdiff(img, refimg_resized)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    
    # YOLO Detection
    model = YOLO(YOLO_WEIGHTS)
    results = model(diff, device="cpu", conf=0.3, verbose=False)
    
    # Score berechnen
    best_tip, center_x, center_y = get_best_dart_tip(results)
    detected_point = [center_x, center_y] if best_tip else None
    points, str_points = get_score_from_tip(best_tip, H)
    
    return points, str_points, diff, diff_thresh
