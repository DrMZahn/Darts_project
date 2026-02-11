import tkinter as tk
from tkinter import messagebox
import cv2
import threading
import time
import json
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
from pathlib import Path

# Lokale Imports
from src.config.constants import *
from src.camera.webcam import WebcamManager
from src.calibration.manual_calib import calibrate_dartboard_manual
from src.calibration.auto_calib import auto_calibrate_dartboard
from src.detection.yolo_detection import analyze_single_dart
from src.utils.image_utils import save_numbered_images_with_reference, get_next_filename
from src.scoring.scoring_logic import compute_score_from_tip

class DartsApp:
    def __init__(self):
        self.root = tk.Tk()
        self.webcam = WebcamManager()
        
        # Globale State Variablen
        self.H = None
        self.refimg = None
        self.refimg_filename = None
        self.stream_running = False
        self.auto_mode_active = False
        self.last_detection_time = 0
        
        # Spiel State
        self.player_names = ["Papa", "Erik"]
        self.player_scores = [301, 301]
        self.current_player = 0
        self.game_round_counter = 1
        self.dart_counter_in_group = 0
        self.round_points = []
        self.str_round_points = []
        self.throws_in_group = 0
        
        # GUI Elements
        self.video_label = None
        self.status_label = None
        self.throws_label = None
        self.points_detail_label = None
        self.counter_label = None
        self.lbl_p1_var = None
        self.lbl_p2_var = None
        self.lbl_current_var = None
        self.btn_stream = None
        self.btn_auto_toggle = None
        
        self.setup_gui()
        self.update_display()
    
    def setup_gui(self):
        """GUI mit DEINEN ORIGINAL-TEXTEN"""
        self.root.title("Darts Scoring - AUTO-DETECTION 🔥")
        self.root.geometry("1920x1080")

        # DEIN Original-Status
        self.status_label = tk.Label(self.root, text="Darts-System | REFERENZ + 3 Würfe pro Runde", font=("Arial", 14), bg="lightgreen")
        self.status_label.pack(pady=10)

        # DEINE Statuszeile
        status_counter_frame = tk.Frame(self.root, bg='lightgray')
        status_counter_frame.pack(fill=tk.X, padx=10, pady=5)

        tk.Label(status_counter_frame, text="Runde:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.counter_label = tk.Label(status_counter_frame, text=f"{self.game_round_counter:03d}", font=("Arial", 16, "bold"), bg="yellow", width=6)
        self.counter_label.pack(side=tk.LEFT, padx=5)

        tk.Label(status_counter_frame, text="| Würfe:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
        self.throws_label = tk.Label(status_counter_frame, text="0/3", font=("Arial", 16, "bold"), bg="lightblue", width=6)
        self.throws_label.pack(side=tk.LEFT, padx=5)

        tk.Label(status_counter_frame, text="|", font=("Arial", 12)).pack(side=tk.LEFT)
        self.points_detail_label = tk.Label(status_counter_frame, text=f"{self.player_names[self.current_player]}, STARTE SPIEL", font=("Arial", 14, "bold"), bg="lightgreen", width=30)
        self.points_detail_label.pack(side=tk.LEFT, padx=5)

        # Video Frame (DEIN Layout)
        video_frame = tk.Frame(self.root, bg='gray', width=1000, height=600)
        video_frame.pack(side=tk.LEFT, padx=10, pady=10)
        video_frame.pack_propagate(False)
        self.video_label = tk.Label(video_frame, bg="black", text="Kamera-Bild erscheint hier")
        self.video_label.pack(expand=True)

        # DEINE ORIGINAL BUTTONS - EXAKT so!
        control_frame = tk.Frame(self.root, bg='lightgray', width=350)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        control_frame.pack_propagate(False)

        # DEINE Button-Texte:
        btn_calib = tk.Button(control_frame, text="Manuell kalibrieren", command=self.on_manual_calib, bg="gainsboro", fg="white", font=("Arial", 11, "bold"), height=2)
        btn_calib.pack(fill=tk.X, pady=3)

        btn_auto_calib = tk.Button(control_frame, text="Automatisch kalibrieren", command=self.on_auto_calib, bg="lightgreen", fg="white", font=("Arial", 11, "bold"), height=2)
        btn_auto_calib.pack(fill=tk.X, pady=3)

        self.btn_stream = tk.Button(control_frame, text="Stream START", command=self.toggle_stream, bg="lightgreen", font=("Arial", 12, "bold"), height=2)
        self.btn_stream.pack(fill=tk.X, pady=5)

        self.btn_auto_toggle = tk.Button(control_frame, text="Auto mode: AUS", command=self.toggle_auto_mode, bg="gainsboro", fg="white", font=("Arial", 11), height=2)
        self.btn_auto_toggle.pack(fill=tk.X, pady=5)

        btn_ref = tk.Button(control_frame, text="Referenzbild aufnehmen ", command=self.on_capture_reference, bg="lightcoral", font=("Arial", 14, "bold"), height=4)
        btn_ref.pack(fill=tk.X, pady=5)

        btn_dart = tk.Button(control_frame, text="Detektiere Pfeil manuell", command=self.on_analyze_one_dart, bg="lightcoral", font=("Arial", 14, "bold"), height=4)
        btn_dart.pack(fill=tk.X, pady=5)
        

        # DEINE Score-Anzeige
        score_frame = tk.LabelFrame(control_frame, text="Scores (301)", font=("Arial", 12))
        score_frame.pack(fill=tk.X, pady=10)

        self.lbl_p1_var = tk.StringVar(value=f"{self.player_names[0]}: {self.player_scores[0]}")
        self.lbl_p2_var = tk.StringVar(value=f"{self.player_names[1]}: {self.player_scores[1]}")
        self.lbl_current_var = tk.StringVar(value=f"Am Pfeil: {self.player_names[self.current_player]}")

        tk.Label(score_frame, textvariable=self.lbl_p1_var, font=("Arial", 20, "bold")).pack(pady=10)
        tk.Label(score_frame, textvariable=self.lbl_p2_var, font=("Arial", 20, "bold")).pack(pady=10)
        tk.Label(score_frame, textvariable=self.lbl_current_var, font=("Arial", 16, "bold")).pack(pady=10)    

        btn_reset = tk.Button(control_frame, text="Neues Spiel (Reset)", 
                  command=self.reset_game, bg="lightblue", height=2)
        btn_reset.pack(fill=tk.X, pady=3)


    
    
    def update_display(self):
        """Aktualisiert Score-Anzeigen"""
        self.lbl_p1_var.set(f"{self.player_names[0]}: {self.player_scores[0]}")
        self.lbl_p2_var.set(f"{self.player_names[1]}: {self.player_scores[1]}")
        self.lbl_current_var.set(f"Am Zug: {self.player_names[self.current_player]}")
        self.counter_label.config(text=f"{self.game_round_counter:03d}")
    
    def load_calibration(self) -> bool:
        """Lädt Kalibrierung aus JSON"""
        if not Path(JSON_PATH).exists():
            messagebox.showerror("Fehler", f"JSON nicht gefunden: {JSON_PATH}")
            return False
        
        with open(JSON_PATH, "r") as f:
            data = json.load(f)
        pts = data["points"]
        
        src_pts = np.array([[pts[name]["x"] * scale_x, pts[name]["y"] * scale_y] 
                           for name in POINT_NAMES[1:]], dtype=np.float32)
        dst_pts = np.array([[CENTER, CENTER - RADIUS], [CENTER + RADIUS, CENTER], 
                           [CENTER, CENTER + RADIUS], [CENTER - RADIUS, CENTER]], dtype=np.float32)
        
        self.H, _ = cv2.findHomography(src_pts, dst_pts)
        print("✅ Homographie Matrix geladen")
        self.status_label.config(text="✅ Kalibrierung geladen!", bg="lightgreen")
        return True
    
    # === EVENT HANDLER ===
    def on_manual_calib(self):
        """Manuelle Kalibrierung starten"""
        if calibrate_dartboard_manual(self.webcam, self.game_round_counter):
            self.load_calibration()
    
    def on_auto_calib(self):
        """Automatische YOLO-Kalibrierung"""
        if auto_calibrate_dartboard(self.webcam, self.game_round_counter):
            self.load_calibration()
    
    def toggle_stream(self):
        """Stream ein/aus"""
        if not self.stream_running:
            self.start_stream()
        else:
            self.stop_stream()
    
    def start_stream(self):
        """Stream starten"""
        if self.webcam.init_camera():
            self.stream_running = True
            self.btn_stream.config(text="⏹️ Stream STOP", bg="red")
            self.update_stream()
        else:
            messagebox.showerror("Fehler", f"Kamera /dev/video{CAM_INDEX} nicht verfügbar!")
    
    def stop_stream(self):
        """Stream stoppen"""
        self.stream_running = False
        self.btn_stream.config(text="📹 Stream START", bg="lightgreen")
    
    def update_stream(self):
        """Stream Update Loop mit Auto-Detection"""
        if self.stream_running and self.webcam.cap and self.webcam.cap.isOpened():
            ret, frame = self.webcam.cap.read()
            if ret:
                frame_display = cv2.resize(frame, (854, 480))
                
                # Kalibrierungspunkte overlay
                if self.H is not None and Path(JSON_PATH).exists():
                    try:
                        with open(JSON_PATH, "r") as f:
                            data = json.load(f)
                        pts = data["points"]
                        scale_factor = frame_display.shape[1] / ORIG_W
                        
                        for i, name in enumerate(POINT_NAMES[1:]):
                            if name in pts:
                                x = int(pts[name]["x"] * scale_factor)
                                y = int(pts[name]["y"] * scale_factor)
                                colors = [(0, 255, 255), (255, 255, 255), (0,255,0), (255,125,0)]
                                color = colors[i % len(colors)]
                                
                                cv2.circle(frame_display, (x, y), 8, color, 2)
                                cv2.putText(frame_display, name[:4], (x+10, y-5), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    except:
                        pass
                
                # Auto-Detection
                if self.auto_mode_active and self.refimg is not None:
                    try:
                        h, w = frame.shape[:2]
                        ref_resized = cv2.resize(self.refimg, (w, h))
                        diff = cv2.absdiff(frame, ref_resized)
                        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                        _, diff_thresh = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
                        
                        change_pixels = np.sum(diff_thresh == 255)
                        current_time = time.time()
                        
                        if change_pixels > 8000 and (current_time - self.last_detection_time) > 1.5:
                            print(f"🎯 AUTO-DETECTED! Pixel: {change_pixels}")
                            self.last_detection_time = current_time
                            self.on_analyze_one_dart(auto_trigger=True)
                    except:
                        pass
                
                # Auto-Status Overlay
                auto_text = "AUTO ON" if self.auto_mode_active else "AUTO OFF"
                cv2.rectangle(frame_display, (10, frame_display.shape[0]-40), (200, frame_display.shape[0]), (0,0,0), -1)
                cv2.putText(frame_display, auto_text, (15, frame_display.shape[0]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Anzeige
                frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
        
        if self.stream_running:
            self.video_label.after(50, self.update_stream)
    
    def toggle_auto_mode(self):
        """Auto-Detection ein/aus"""
        self.auto_mode_active = not self.auto_mode_active
        self.btn_auto_toggle.config(text=f"AUTO: {'ON' if self.auto_mode_active else 'OFF'}", 
                                  bg="lightpink" if self.auto_mode_active else "gainsboro")
        status = "🔥 AUTO AKTIV!" if self.auto_mode_active else "⏹️ AUTO DEAKTIV!"
        print(status)
    
    def on_capture_reference(self):
        """Referenzbild aufnehmen"""
        try:
            frame = self.webcam.capture_frame()
            self.refimg_filename = get_next_filename("ref", self.game_round_counter, self.current_player)
            self.refimg = frame.copy()
            
            Path("current_game").mkdir(exist_ok=True)
            cv2.imwrite(f"current_game/{self.refimg_filename}", self.refimg)
            
            self.dart_counter_in_group = 0
            self.round_points = []
            self.throws_label.config(text="0/3")
            self.points_detail_label.config(text="PTS: ---", bg="lightgray")
            self.status_label.config(text=f"Referenz aufgenommen, {self.player_names[self.current_player]} kann werfen, Runde {self.game_round_counter:03d}", bg="lightgreen")
            self.update_display()
            
            # Auto-Mode automatisch starten
            if not self.auto_mode_active:
                self.toggle_auto_mode()
                
        except Exception as e:
            self.status_label.config(text=f"❌ Referenz-Fehler: {str(e)}", bg="red")
    
    def on_analyze_one_dart(self, auto_trigger=False):
        """Komplette Funktion - FEHLERFREI"""
        
        if self.refimg is None: 
            self.status_label.config(text="❌ Zuerst REFERENZ aufnehmen!", bg="red")
            self.root.after(2000, lambda: self.status_label.config(text="Darts-System | REFERENZ + 3 Würfe pro Runde", bg="lightgreen"))
            return
        
        if self.H is None:
            if not self.load_calibration():
                self.status_label.config(text="❌ Zuerst KALIBRIEREN!", bg="red")
                return
        
        try:
            img = self.webcam.capture_frame()
            self.dart_counter_in_group += 1
            self.throws_label.config(text=f"{self.dart_counter_in_group}/3")
            
            # Diff-Berechnung
            h, w = img.shape[:2]
            refimg_resized = cv2.resize(self.refimg, (w, h))
            diff = cv2.absdiff(img, refimg_resized)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, diff_thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
            
            # YOLO Detection
            model = YOLO(YOLO_WEIGHTS)
            results = model(diff, device="cpu", conf=0.3, verbose=False)
            
            # Beste Dartspitze finden
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
                        break
                    break
            
            detected_point = [center_x, center_y] if best_tip else None
            
            # **TRANSFORM + SCORE** (DEIN Original)
            if best_tip:
                point_array = np.array([[[float(best_tip[0]), float(best_tip[1])]]], dtype=np.float32)
                warped = cv2.perspectiveTransform(point_array, self.H)
                exact_tip = (int(warped[0, 0, 0]), int(warped[0, 0, 1]))
                points, str_points = compute_score_from_tip(exact_tip)
            else:
                points, str_points = 0, " 0"
            
            print(f"📍 Tip: {best_tip} → {points} {str_points}")
            
            # **BILDER SPEICHERN** - KORREKTE Parameter!
            save_numbered_images_with_reference(
                img, 
                detected_point=detected_point, 
                current_player=self.current_player, 
                score=str_points, 
                diff=diff, 
                diff_thresh=diff_thresh,
                game_round=self.game_round_counter  # ✅ Nur diese Parameter!
            )
            
            # **DOUBLE OUT LOGIK** 🔥
            old_score = self.player_scores[self.current_player]
            is_double = "D" in str_points or str_points in ["B25", "B50"]
            
            # Runden-Punkte
            self.round_points.append(points)
            self.str_round_points.append(str_points)
            total_round = sum(self.round_points)
            
            if self.dart_counter_in_group == 1:
                points_text = f"PTS: {str_points} => {total_round}"
            else:
                points_text = f"PTS: {self.str_round_points[0]} + {str_points} => {total_round}"
            
            # **DOUBLE OUT CHECK**
            if old_score <= points:
                if points == old_score and is_double:
                    # 🎉 GEWONNEN!
                    self.player_scores[self.current_player] = 0
                    points_text = f"🎉 DOUBLE OUT! {str_points} -> GEWONNEN!"
                    self.points_detail_label.config(text=points_text, bg="gold")
                    self.status_label.config(text=f"🏆 {self.player_names[self.current_player]} GEWONNEN!", bg="gold")
                    self.root.after(2000, lambda: messagebox.showinfo("🏆 SIEGER!", f"{self.player_names[self.current_player]} gewinnt mit Double {str_points}!"))
                else:
                    # 💥 BUST!
                    self.round_points.pop()
                    self.str_round_points.pop()
                    points_text = f"💥 BUST! ({str_points}) -> Score bleibt: {old_score}"
                    self.points_detail_label.config(text=points_text, bg="red")
                    self.status_label.config(text="BUST! Nächster Wurf...", bg="orange")
            else:
                # ✅ Normaler Wurf
                new_score = max(0, old_score - points)
                self.player_scores[self.current_player] = new_score
                self.points_detail_label.config(text=points_text, bg="lightgreen")
            
            # Scores aktualisieren
            self.lbl_p1_var.set(f"{self.player_names[0]}: {self.player_scores[0]}")
            self.lbl_p2_var.set(f"{self.player_names[1]}: {self.player_scores[1]}")
            self.lbl_current_var.set(f"Am Pfeil: {self.player_names[self.current_player]}")
            
            # Neue Referenz
            self.refimg = img
    
            # Runde beenden?
            if self.dart_counter_in_group >= 3:
                next_player = 1 - self.current_player
                self.lbl_current_var.set(f"Nächster Spieler: {self.player_names[next_player]}")
                
                points_list_text = " + ".join(map(str, self.str_round_points)) + f" => {sum(self.round_points)}"
                self.points_detail_label.config(text=f"Beendet: {points_list_text} PTS!", bg="gold")
                self.throws_label.config(text="3/3")
                self.status_label.config(text=f"{self.player_names[self.current_player]} hat geworfen! Pfeile raus. Dann ist {self.player_names[next_player]} dran. Aber erst neue Referenz aufnehmen! ...", bg="lightblue")
                
                # Auto-Mode off
                self.auto_mode_active = False
                self.refimg = None
                
                if self.current_player == 1:
                    self.game_round_counter += 1
                    self.counter_label.config(text=f"{self.game_round_counter:03d}")
                
                self.current_player = next_player
                self.round_points = []
                self.str_round_points = []
                    
        except Exception as e:
            print(f"on_analyze_one_dart EXCEPTION: {e}")
            self.status_label.config(text=f"ANALYSE-FEHLER: {str(e)}", bg="red")

    def end_round(self):
        """Runde beenden, nächsten Spieler"""
        next_player = 1 - self.current_player
        points_list_text = " + ".join(map(str, self.str_round_points)) + f" => {sum(self.round_points)}"
        
        self.points_detail_label.config(text=f"Runde OK: {points_list_text}", bg="gold")
        self.throws_label.config(text="3/3")
        
        self.lbl_current_var.set(f"Nächster: {self.player_names[next_player]}")
        self.status_label.config(text="Pfeile rausziehen → NEUE REFERENZ!", bg="lightblue")
        
        # Auto-Mode aus
        self.auto_mode_active = False
        self.refimg = None
        
        if self.current_player == 1:  # Letzter Spieler → neue Runde
            self.game_round_counter += 1
            self.status_label.config(text="🎉 NEUE RUNDE! REFERENZ aufnehmen", bg="gold")
        
        self.current_player = next_player
        self.round_points = []
        self.str_round_points = []
        self.dart_counter_in_group = 0
    
    def run(self):
        """Hauptschleife starten"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.mainloop()
    
    def on_close(self):
        """Cleanup"""
        self.stream_running = False
        self.webcam.release()
        cv2.destroyAllWindows()
        self.root.destroy()

    def reset_game(self):
        """Komplett neues Spiel starten"""
        self.player_scores = [301, 301]
        self.current_player = 0
        self.game_round_counter = 1
        self.dart_counter_in_group = 0
        self.round_points = []
        self.str_round_points = []
        self.refimg = None
        self.H = None
        
        self.throws_label.config(text="0/3")
        self.points_detail_label.config(text="Neues Spiel gestartet!", bg="lightgreen")
        self.update_display()
        self.status_label.config(text="Neues Spiel | REFERENZ + 3 Würfe pro Runde", bg="lightgreen")


if __name__ == "__main__":
    app = DartsApp()
    app.run()
