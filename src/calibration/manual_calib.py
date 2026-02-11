import cv2
import json
import os
import tkinter as tk
from tkinter import messagebox
from typing import Dict, Any

# Nur echte Konstanten importieren
from src.config.constants import JSON_PATH, POINT_NAMES

# Globale Kalibrierungs-Variablen (nur lokal)
clicked_points: Dict[str, Dict[str, int]] = {}
current_index = 0
calib_img = None
current_index = 0  # Global für mouse_callback

def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
    global current_index, calib_img, clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and current_index < len(POINT_NAMES):
        name = POINT_NAMES[current_index]
        clicked_points[name] = {"x": x, "y": y}

        cv2.circle(calib_img, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(calib_img, name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 0), 1, cv2.LINE_AA)
        current_index += 1
        cv2.imshow("KALIBRIERUNG - Dartscheibe (ESC/Q Ende)", calib_img)

        if current_index == len(POINT_NAMES):
            print("Alle Punkte gesetzt – ESC oder Q zum Beenden")

def calibrate_dartboard_manual(webcam, game_round_counter: int, game_folder: str = "current_game") -> bool:
    """Manuelle Kalibrierung durch Klick auf 5 Punkte"""
    global calib_img, clicked_points, current_index
    
    try:
        frame = webcam.capture_frame()
        calib_img = frame.copy()
        num = f"{game_round_counter:03d}"
        
        os.makedirs(game_folder, exist_ok=True)
        cv2.imwrite(os.path.join(game_folder, f"calib_img_ref_{num}.jpg"), calib_img)
        
        clicked_points.clear()
        current_index = 0
        
        cv2.namedWindow("KALIBRIERUNG - Dartscheibe (ESC/Q Ende)", cv2.WINDOW_NORMAL)
        cv2.imshow("KALIBRIERUNG - Dartscheibe (ESC/Q Ende)", calib_img)
        cv2.setMouseCallback("KALIBRIERUNG - Dartscheibe (ESC/Q Ende)", mouse_callback)
        
        print("🔧 KALIBRIERUNG: Klicke 5 Punkte auf der Dartscheibe:")
        for i, name in enumerate(POINT_NAMES):
            print(f"   {i+1}. {name}")
        
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord('q')):
                break
            if cv2.getWindowProperty("KALIBRIERUNG - Dartscheibe (ESC/Q Ende)", 
                                   cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cv2.destroyAllWindows()
        
        if len(clicked_points) != len(POINT_NAMES):
            print("❌ Abgebrochen: nicht alle Punkte gesetzt")
            messagebox.showwarning("Kalibrierung", "Nicht alle Punkte gesetzt!")
            return False
        
        data = {
            "image": f"calib_img_ref_{num}.jpg",
            "refimage": f"ref_empty_throw_{num}.jpg",
            "points": clicked_points
        }
        
        with open(JSON_PATH, "w") as f:
            json.dump(data, f, indent=4)
        
        print(f"✅ KALIBRIERUNG GESPEICHERT: {JSON_PATH}")
        messagebox.showinfo("Kalibrierung", f"Punkte gespeichert!\n{JSON_PATH}")
        return True
        
    except Exception as e:
        messagebox.showerror("Kalibrierungsfehler", str(e))
        return False
