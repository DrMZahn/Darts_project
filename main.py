import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import sys
import os

# Module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game.logic import Game
# from vision.capture import capture_board_frame
from vision.nn_inference import DartsNetWrapper
from vision.detrect_score import detect_dart_score_visual


class DartsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Dart Scorer - 1080P FHD Camera + FELDER")
        self.root.geometry("1600x900")
        self.game = None
        self.net = None
        self.cap = None
        self.cameras = []
        self.selected_cam = 2
        self.live_image = None
        self.photo_image = None
        self.score_labels = {}
        self.player_frames = {}
        self.full_frame = None
        self.frame_count = 0

        # Feste ROI (x, y, w, h) – an deine Kamera/Dartscheibe anpassen
        self.roi = (640, 120, 2500, 2500)

        self.setup_ui()
        self.scan_cameras()

    def detect_dartboard_adaptive_fast(self, frame_bgr):
        """
        ⚡ SCHNELLE VERSION: Nur 1 Scale + optimierte Filter
        Erkennt trotzdem ALLE wichtigen Flächen!
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # 1. SCHNELLES K-MEANS (weniger Cluster, weniger Iterationen)
        data = hsv.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, _ = cv2.kmeans(data, 6, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
        
        frame_result = frame_bgr.copy()
        valid_regions = 0
        
        # 2. NUR 6 CLUSTER statt 12 (schneller)
        for i in range(6):
            mask = (labels.reshape(frame_bgr.shape[:2]) == i).astype(np.uint8) * 255
            
            # SEHR KLEINER Kernel für Speed
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # SEHR LOCKERE Filter für MAXIMUM Erkennung
                if 200 < area < frame_bgr.shape[0]*frame_bgr.shape[1]*0.3:
                    # Farbe nach Cluster-Index
                    colors = [(0,255,0), (0,255,255), (255,0,255), (255,255,0), (255,0,0), (0,0,255)]
                    color = colors[i % 6]
                    
                    cv2.drawContours(frame_result, [contour], -1, color, 2)
                    
                    # Schneller Mittelpunkt
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(frame_result, (cx, cy), 4, color, -1)
                        
                        # KURZES Label
                        cv2.putText(frame_result, f"{int(area/1000)}k", 
                                   (cx+8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        valid_regions += 1
        
        cv2.putText(frame_result, f"⚡ {valid_regions} Flächen", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame_result
    
    def detect_dartboard_colors_only(self, frame_bgr):
        """
        🎯 OPTIMIERTE VERSION: NUR ROT, GRÜN, SCHWARZ, BEIGE
        Sehr schnell + präzise für Dartsscheibe!
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        frame_result = frame_bgr.copy()
        valid_regions = 0
        
        # 1. DEFINIERTE 4 DARTSCHEIBEN-FARBEN (HSV-Bereiche)
        color_ranges = {
            'ROT': ([191, 97, 45], [211, 117, 65]),
            'GRUEN': ([0, 102, 37], [13, 122, 57]), 
            'SCHWARZ': ([10, 20, 10], [30, 50, 30]),      # Dunkelgrau/Schwarz
            'BEIGE': ([8, 25, 9], [28, 45, 29])     # Beige/Hellbraun
        }
        
        colors_bgr = {
            'ROT': (201, 107, 55),
            'GRUEN': (3, 112, 47), 
            'SCHWARZ': (20, 30, 20),
            'BEIGE': (18, 35, 19)
        }
        
        # 2. JEDE FARBKLASSE extrahieren
        for color_name, (lower, upper) in color_ranges.items():
            # Präzise Maske
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Leicht glätten
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
            
            # Konturen finden
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Dartsscheiben-typische Größe
                if 500 < area < frame_bgr.shape[0]*frame_bgr.shape[1]*0.2:
                    color_bgr = colors_bgr[color_name]
                    
                    # Fläche markieren
                    cv2.drawContours(frame_result, [contour], -1, color_bgr, 3)
                    
                    # Zentrum + Label
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.circle(frame_result, (cx, cy), 6, color_bgr, -1)
                        
                        # Farbname + Größe
                        cv2.putText(frame_result, f"{color_name}:{int(area/1000)}k", 
                                   (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
                        valid_regions += 1
        
        # Status
        cv2.putText(frame_result, f"🎯 {valid_regions} Farbflächen (R,G,S,B)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame_result

    def setup_ui(self):
        # HEADER
        header = tk.Frame(self.root, bg="#2196F3")
        header.pack(fill=tk.X, pady=5)
        tk.Label(
            header,
            text="🎯 DART SCORER - 1080P FHD Camera + FELDER DETEKTION",
            font=("Arial", 22, "bold"),
            bg="#2196F3",
            fg="white",
        ).pack(pady=10)

        # KAMERA-AUSWAHL
        cam_frame = tk.Frame(header)
        cam_frame.pack(pady=5)
        tk.Label(
            cam_frame,
            text="🎥 Kamera:",
            font=("Arial", 14),
            bg="#2196F3",
            fg="white",
        ).pack(side=tk.LEFT)

        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(
            cam_frame, textvariable=self.camera_var, width=35, state="readonly"
        )
        self.camera_combo.pack(side=tk.LEFT, padx=10)
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_select)

        tk.Button(
            cam_frame,
            text="🔍 Refresh",
            command=self.scan_cameras,
            bg="#FF9800",
            fg="white",
            width=10,
        ).pack(side=tk.LEFT, padx=5)
        self.cam_status = tk.Label(
            cam_frame, text="Scanne...", fg="orange", bg="#2196F3"
        )
        self.cam_status.pack(side=tk.LEFT, padx=10)

        # SPIELER
        player_frame = tk.Frame(self.root)
        player_frame.pack(pady=20)
        tk.Label(player_frame, text="Spieler 1:", font=("Arial", 12)).grid(
            row=0, column=0, padx=10
        )
        self.p1 = tk.Entry(player_frame, width=12, font=("Arial", 12))
        self.p1.grid(row=0, column=1, padx=5)
        self.p1.insert(0, "Spieler 1")

        tk.Label(player_frame, text="Spieler 2:", font=("Arial", 12)).grid(
            row=0, column=2, padx=10
        )
        self.p2 = tk.Entry(player_frame, width=12, font=("Arial", 12))
        self.p2.grid(row=0, column=3, padx=5)
        self.p2.insert(0, "Spieler 2")

        tk.Button(
            player_frame,
            text="▶️ SPIEL STARTEN",
            command=self.start_game,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 16, "bold"),
            width=15,
            height=2,
        ).grid(row=1, column=0, columnspan=4, pady=15)

        # HAUPTBEREICH
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    def scan_cameras(self):
        self.cameras = []
        print("\n🎥 === 1080P FHD PRIORISIERT ===")

        for i in [2, 3]:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                ret, frame = cap.read()
                if ret:
                    w, h = frame.shape[1], frame.shape[0]
                else:
                    w, h = 640, 480
                cap.release()
                cam_name = f"🎥 1080P FHD (/dev/video{i}) ({w}x{h}px)"
                self.cameras.append((i, cam_name))
                print(f"✓ PRIORITÄT {cam_name}")

        for i in [0, 1]:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                ret, frame = cap.read()
                if ret:
                    w, h = frame.shape[1], frame.shape[0]
                else:
                    w, h = 640, 480
                cap.release()
                cam_name = f"🎥 Chicony USB 2.0 (/dev/video{i}) ({w}x{h}px)"
                self.cameras.append((i, cam_name))
                print(f"✓ {cam_name}")

        if self.cameras:
            names = [name for _, name in self.cameras]
            self.camera_combo["values"] = names
            for idx, (_, name) in enumerate(self.cameras):
                if "1080P FHD" in name:
                    self.camera_combo.current(idx)
                    self.selected_cam = self.cameras[idx][0]
                    break
            else:
                self.camera_combo.current(0)
                self.selected_cam = self.cameras[0][0]

            self.camera_var.set(self.camera_combo.get())
            self.cam_status.config(
                text=f"✓ {len(self.cameras)} Kameras - 1080P FHD priorisiert",
                fg="green",
            )
        else:
            self.camera_combo["values"] = ["Keine Kamera"]
            self.cam_status.config(text="❌ Keine Kameras!", fg="red")

    def on_camera_select(self, event):
        idx = self.camera_combo.current()
        if idx >= 0:
            self.selected_cam = self.cameras[idx][0]
            if hasattr(self, "status_label"):
                self.status_label.config(
                    text=f"Kamera: {self.camera_var.get()}", fg="blue"
                )

    def start_game(self):
        if not self.cameras:
            messagebox.showerror("❌ Fehler", "Keine 1080P FHD Kamera gefunden!")
            return
        self.game = Game([self.p1.get(), self.p2.get()])
        self.net = DartsNetWrapper("dummy")
        self.build_game_ui()

    def build_game_ui(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # LINKS: SCORES
        left_frame = tk.Frame(
            self.main_frame, bg="#e0e0e0", relief="ridge", bd=3, width=350
        )
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_frame.pack_propagate(False)

        tk.Label(
            left_frame, text="📊 SCORES", font=("Arial", 18, "bold"), bg="#e0e0e0"
        ).pack(pady=15)
        self.score_labels = {}
        self.player_frames = {}

        for player in self.game.players:
            p_frame = tk.Frame(left_frame, bg="white", relief="ridge", bd=2)
            p_frame.pack(pady=15, padx=20, fill=tk.X)
            name_lbl = tk.Label(p_frame, text=player.name, font=("Arial", 14, "bold"), bg="white")
            name_lbl.pack(pady=5)
            score_lbl = tk.Label(
                p_frame,
                text=str(player.score),
                font=("Arial", 36, "bold"),
                fg="#d32f2f",
                bg="white",
            )
            score_lbl.pack(pady=10)
            self.score_labels[player.name] = score_lbl
            self.player_frames[player.name] = p_frame

        self.turn_label = tk.Label(
            left_frame,
            text=f"🎯 {self.game.current_player.name}",
            font=("Arial", 16, "bold"),
            fg="#388e3c",
            bg="#e0e0e0",
        )
        self.turn_label.pack(pady=25)

        self.status_label = tk.Label(
            left_frame,
            text=f"1080P FHD Kamera: /dev/video{self.selected_cam}",
            font=("Arial", 12),
            fg="#1976D2",
            bg="#e0e0e0",
        )
        self.status_label.pack(pady=10)

        # MITTE: LIVE 1080P FHD
        self.live_frame = tk.Frame(self.main_frame, relief="ridge", bd=3, bg="black")
        self.live_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))
        self.live_frame.pack_propagate(False)

        tk.Label(
            self.live_frame,
            text="🔴 LIVE 1080P FHD WEBCAM",
            font=("Arial", 16, "bold"),
            bg="#D32F2F",
            fg="white",
        ).pack(pady=10)
        self.live_label = tk.Label(
            self.live_frame, bg="#424242", text="▶️ LIVE STARTEN klicken"
        )
        self.live_label.pack(fill=tk.BOTH, expand=True, pady=15)
        tk.Button(
            self.live_frame,
            text="▶️ LIVE STARTEN",
            command=self.start_live,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 14, "bold"),
            width=15,
            height=2,
        ).pack(pady=10)

        # RECHTS: FOTO 1080P FHD
        self.photo_frame = tk.Frame(self.main_frame, relief="ridge", bd=3, bg="black")
        self.photo_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.photo_frame.pack_propagate(False)

        tk.Label(
            self.photo_frame,
            text="📸 FOTO 1080P FHD + ALLE 20 FELDER + Ringe",
            font=("Arial", 16, "bold"),
            bg="#FF9800",
            fg="white",
        ).pack(pady=10)
        self.image_label = tk.Label(
            self.photo_frame, bg="#616161", text="Foto erscheint hier..."
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=15)

        # BUTTONS
        self.btn_frame = tk.Frame(self.root, bg="lightgray")
        self.btn_frame.pack(fill=tk.X, pady=15)
        tk.Button(
            self.btn_frame,
            text="⏹ LIVE STOP",
            command=self.stop_live,
            bg="#FF9800",
            fg="white",
            font=("Arial", 12, "bold"),
            width=14,
            height=2,
        ).pack(side=tk.LEFT, padx=15)
        tk.Button(
            self.btn_frame,
            text="💾 JPG SPEICHERN", 
            command=self.save_debug_jpg,
            bg="#2196F3", 
            fg="white", 
            font=("Arial", 14, "bold"), 
            width=16, 
            height=2
        ).pack(side=tk.LEFT, padx=10)        
        self.btn_throw = tk.Button(
            self.btn_frame,
            text="🎯 3 DARTS AUSWERTEN",
            command=self.process_throw,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 16, "bold"),
            width=22,
            height=3,
        )
        self.btn_one_throw = tk.Button(
            self.btn_frame,
            text="🎯 1 DARTS AUSWERTEN",
            command=self.process_one_throw,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 16, "bold"),
            width=22,
            height=3,
        )
        
        self.btn_throw.pack(side=tk.LEFT, padx=20)
        self.btn_one_throw.pack(side=tk.LEFT, padx=20)
        tk.Button(
            self.btn_frame,
            text="❌ BEENDEN",
            command=self.root.quit,
            bg="#F44336",
            fg="white",
            font=("Arial", 12, "bold"),
            width=14,
            height=2,
        ).pack(side=tk.RIGHT, padx=15)

        self.highlight_current_player()

    def highlight_current_player(self):
        for player in self.game.players:
            frame = self.player_frames[player.name]
            if player is self.game.current_player:
                frame.config(bg="#C8E6C9")
                for child in frame.winfo_children():
                    child.config(bg="#C8E6C9")
            else:
                frame.config(bg="white")
                for child in frame.winfo_children():
                    child.config(bg="white")

    def start_live(self):
        try:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.selected_cam)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                if not self.cap.isOpened():
                    self.status_label.config(
                        text="❌ 1080P FHD Kamera Fehler!", fg="red"
                    )
                    return
            self.update_live()
            self.status_label.config(
                text=f"✅ LIVE 1080P FHD /dev/video{self.selected_cam} (1920x1080)",
                fg="green",
            )
        except Exception as e:
            self.status_label.config(text=f"❌ Live Fehler: {e}", fg="red")

    def update_live(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.full_frame = frame.copy()  # Originalkamera-Frame speichern
                self.frame_count += 1

                # Größe des Panels abfragen
                panel_width = max(self.live_label.winfo_width(), 900)
                panel_height = max(self.live_label.winfo_height(), 650)

                # Frame für Anzeige im Panel skalieren
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = Image.fromarray(frame_rgb).resize(
                    (panel_width, panel_height), Image.Resampling.LANCZOS
                )

                self.live_image = ImageTk.PhotoImage(frame_resized)
                self.live_label.config(image=self.live_image, text="")

            self.root.after(33, self.update_live)

    def stop_live(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.live_label.config(image="", text="Live gestoppt")
        self.status_label.config(text="Live gestoppt", fg="orange")
        
    def save_debug_jpg(self):
        """Speichert aktuelles Frame als JPG zur Analyse"""
        try:
            if self.cap is None or not self.cap.isOpened():
                self.status_label.config(text="❌ Kamera nicht aktiv!", fg="red")
                return
            
            ret, frame = self.cap.read()
            if not ret:
                self.status_label.config(text="❌ Kein Frame!", fg="red")
                return
            
            # JPG speichern (JPEG Qualität 95%)
            filename = f"images/test/darts_debug_{int(cv2.getTickCount())}.jpg"
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            self.status_label.config(text=f"✅ {filename} gespeichert!", fg="green")
            print(f"💾 JPG gespeichert: {filename}")
            
        except Exception as e:
            self.status_label.config(text=f"❌ Save Fehler: {e}", fg="red")

    def process_throw(self):
        try:
            self.status_label.config(
                text="📷 1080P FHD Foto + FELDER wird gemacht...", fg="orange"
            )
            self.root.update()

            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.selected_cam)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

            ret, frame = self.cap.read()
            if not ret:
                self.status_label.config(text="❌ Kein Frame für Foto!", fg="red")
                return

            self.full_frame = frame.copy()  # Original speichern

            # PANEL Größe
            panel_width = max(self.image_label.winfo_width(), 900)
            panel_height = max(self.image_label.winfo_height(), 650)

            # *** VOLLSTÄNDIGE DARTSSCHEIBEN-FELDER DETEKTION ***
            frame_marked = self.detect_dartboard_adaptive_fast(frame.copy())

            # Anzeige im Panel (skalieren)
            frame_rgb_marked = cv2.cvtColor(frame_marked, cv2.COLOR_BGR2RGB)
            frame_display = Image.fromarray(frame_rgb_marked).resize(
                (panel_width, panel_height), Image.Resampling.LANCZOS
            )
            self.photo_image = ImageTk.PhotoImage(frame_display)
            self.image_label.config(image=self.photo_image, text="")

            # NN auf dem originalen Frame (optional)
            scores = self.net.predict_scores(frame)

            self.game.register_throw(scores)
            self.update_scores()
            self.status_label.config(
                text=f"✅ {scores} Punkte! Summe: {sum(scores)} + ALLE FELDER", fg="green"
            )

            if self.game.finished:
                messagebox.showinfo("🏆 GEWONNEN!", f"{self.game.winner.name} gewinnt!")

            # Debug-JPG speichern
            filename = f"images/test/darts_debug_fields_{int(cv2.getTickCount())}.jpg"
            cv2.imwrite(filename, frame_marked, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"💾 JPG mit Feldern gespeichert: {filename}")

        except Exception as e:
            self.status_label.config(text=f"❌ Foto Fehler: {e}", fg="red")
            

    def process_one_throw(self):
        try:
            self.status_label.config(
                text="📷 1080P FHD Foto – Wurf wird analysiert...", fg="orange"
            )
            self.root.update()
    
            # Kamera initialisieren
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.selected_cam)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
    
            ret, frame = self.cap.read()
            if not ret:
                self.status_label.config(text="❌ Kein Frame empfangen!", fg="red")
                return
    
            self.full_frame = frame.copy()  # original speichern
    
            # *** Dartscore berechnen ***
            import tempfile
            import os
            # from your_module import detect_dart_score  # ggf. richtigen Modulnamen anpassen
    
            # temporäre Datei, weil detect_dart_score aktuell mit Dateipfaden arbeitet
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, frame)
                tmp_filename = tmp_file.name
            
            
            # from vision.detrect_score import detect_dart_score_visual

            try:
                score = detect_dart_score_visual(tmp_filename)
            finally:
                os.remove(tmp_filename)
            
            
            print(score)
    
            # Anzeige vorbereiten
            panel_width = max(self.image_label.winfo_width(), 900)
            panel_height = max(self.image_label.winfo_height(), 650)
    
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_display = Image.fromarray(frame_rgb).resize(
                (panel_width, panel_height), Image.Resampling.LANCZOS
            )
            self.photo_image = ImageTk.PhotoImage(frame_display)
            self.image_label.config(image=self.photo_image, text="")
    
            # Ergebnis übernehmen
            self.game.register_throw([score])
            self.update_scores()
            self.status_label.config(
                text=f"✅ {score} Punkte erkannt!", fg="green"
            )
    
            if self.game.finished:
                messagebox.showinfo("🏆 GEWONNEN!", f"{self.game.winner.name} gewinnt!")
    
            # Debug-Foto speichern
            filename = f"images/test/darts_debug_{int(cv2.getTickCount())}.jpg"
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"💾 Debug-Foto gespeichert: {filename}")
    
        except Exception as e:
            self.status_label.config(text=f"❌ Fehler beim Verarbeiten: {e}", fg="red")


    def update_scores(self):
        for player in self.game.players:
            self.score_labels[player.name].config(text=str(player.score))
        self.turn_label.config(text=f"🎯 {self.game.current_player.name}")
        self.highlight_current_player()


if __name__ == "__main__":
    # Stelle sicher dass images/test Ordner existiert
    os.makedirs("images/test", exist_ok=True)
    
    root = tk.Tk()
    app = DartsApp(root)
    root.mainloop()
