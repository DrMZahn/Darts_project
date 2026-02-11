import cv2
import json
import os
import numpy as np
from ultralytics import YOLO
from tkinter import messagebox
from typing import Dict, Any

# Nur echte Konstanten
from src.config.constants import (JSON_PATH, YOLO_MODEL_BOARD_PATH, POINT_NAMES)

def auto_calibrate_dartboard(webcam, game_round_counter: int, game_folder: str = "current_game") -> bool:
    """YOLO-basierte automatische Kalibrierung"""
    try:
        print("📸 Webcam-Foto...")
        frame = webcam.capture_frame()
        num = f"{game_round_counter:03d}"
        calib_img_path = os.path.join(game_folder, f"auto_calib_img_{num}.jpg")
        
        os.makedirs(game_folder, exist_ok=True)
        cv2.imwrite(calib_img_path, frame)
        print(f"💾 {calib_img_path}")
        
        print("🤖 YOLO erkennt Randpunkte...")
        model = YOLO(YOLO_MODEL_BOARD_PATH)
        results = model(frame, conf=0.5, verbose=False, imgsz=1024)
        result = results[0]
        
        point_names = ["P_05_20", "P_13_06", "P_17_03", "P_08_11"]
        clicked_points: Dict[str, Dict] = {}
        
        if result.boxes is not None and len(result.boxes) > 0:
            detections = []
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                detections.append({
                    'class': cls, 'conf': conf, 'x': center_x, 'y': center_y
                })
            
            detections.sort(key=lambda d: d['class'])
            
            for i, detection in enumerate(detections[:4]):
                if i < len(point_names):
                    name = point_names[i]
                    clicked_points[name] = {
                        "x": detection['x'], "y": detection['y'], "conf": detection['conf']
                    }
                    print(f"✅ {name}: ({detection['x']}, {detection['y']}) conf={detection['conf']:.2f}")
        
        if len(clicked_points) < 4:
            print(f"⚠️ YOLO fand nur {len(clicked_points)}/4 Punkte")
            messagebox.showwarning("YOLO", f"Nur {len(clicked_points)}/4 Punkte erkannt!")
            return False
        
        # Visualisierung
        viz_img = frame.copy()
        colors = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 165, 0)]
        
        for name, pt in clicked_points.items():
            x, y = pt["x"], pt["y"]
            conf = pt.get("conf", 0)
            color_idx = point_names.index(name)
            color = colors[color_idx]
            
            cv2.circle(viz_img, (x, y), 15, color, -1)
            cv2.circle(viz_img, (x, y), 20, color, 3)
            cv2.putText(viz_img, f"{name} ({conf:.1f})", (x+20, y-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        viz_path = os.path.join(game_folder, f"yolo_calib_viz_{num}.jpg")
        cv2.imwrite(viz_path, viz_img)
        
        data = {
            "image": f"auto_calib_img_{num}.jpg",
            "refimage": f"ref_empty_throw_{num}.jpg",
            "points": clicked_points,
            "auto_detected": True,
            "yolo_detected": True
        }
        
        with open(JSON_PATH, "w") as f:
            json.dump(data, f, indent=4)
        
        print(f"✅ YOLO-KALIBRIERUNG FERTIG: {JSON_PATH}")
        messagebox.showinfo("✅ YOLO Erfolg!", f"{len(clicked_points)}/4 Punkte erkannt!")
        
        cv2.imshow("✅ YOLO AUTO-KALIBRIERUNG", viz_img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO Fehler: {e}")
        messagebox.showerror("YOLO Fehler", str(e))
        return False
