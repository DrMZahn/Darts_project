import cv2  # Stelle sicher dass es ganz oben steht
cv2_module = cv2  # Globale Referenz
import json
import numpy as np
import math
from ultralytics import YOLO
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import time

global throws_label, points_detail_label, counter_label

# -----------------------------
# Pfade / Konstanten - FULL HD!
# -----------------------------
JSON_PATH = "dartboard_points_calibration.json"
#YOLO_WEIGHTS = "/media/PRIV/Darts_ML/runs/detect/runs/darts_gpu_exp02/train/weights/best.pt"
YOLO_WEIGHTS = "/home/matthias/PRIV/Darts_ML/runs/detect/runs/darts_gpu_licht_beide_seiten/train/weights/best.pt"
CAM_INDEX = 2  # /dev/video2

TARGET_SIZE = 2000
CENTER = TARGET_SIZE // 2
RADIUS = 480

R_DOUBLE_OUTER = RADIUS
R_DOUBLE_INNER = int(RADIUS * (160 / 170))
R_TRIPLE_OUTER = int(RADIUS * 110 / 170)
R_TRIPLE_INNER = int(RADIUS * (100 / 170))
R_OUTER_BULL = int(RADIUS * ((31.8 / 2) / 170))
R_INNER_BULL = int(RADIUS * ((14 / 2) / 170))

ORIG_W, ORIG_H = 1920, 1080
NEW_W, NEW_H = 1920, 1080

SEGMENTS = [
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
    3, 19, 7, 16, 8, 11, 14, 9, 12, 5
]

# **ZWEI ZAHLER: Referenz-Gruppe + Bilder innerhalb Gruppe (max 3 Würfe)**
ref_group_counter = 1  # ref_empty_throw_001, ref_empty_throw_002, ...
img_counter = 0        # 0=ref, dann 1st_throw, 2nd_throw, 3rd_throw
throws_in_group = 0    # Zähler für Würfe in aktueller Gruppe (max 3)
dart_counter_in_group = 0  # **NEU: Zählt "1 Dart AUSWERTEN" Klicks**
round_points = []  # **NEU: Punkte dieser Runde [60, 0, 120]**
str_round_points = []
# **GAME_ROUND_COUNTER** (zählt volle Runden nach 2 Spielern)
game_round_counter = 1


# -----------------------------
# Globale Variablen
# -----------------------------
cap = None
refimg = None          
refimg_filename = None 
H = None
scale_x = NEW_W / ORIG_W
scale_y = NEW_H / ORIG_H

# Webcam Stream Variablen
stream_running = False
video_label = None
btn_stream = None

# Scores
player_names=["Papa","Erik"]
player_scores = [301, 301]
current_player = 0

# Kalibrierung Variablen
calib_img = None
calib_window_active = False

# -----------------------------
# **KALIBRIERUNG FUNKTION** (unverändert)
# -----------------------------
POINT_NAMES = ["Mitte", "P_05_20", "P_13_06", "P_17_03", "P_08_11"]
clicked_points = {}
current_index = 0


# autodetect darts thorw
auto_mode_active = False
last_detection_time = 0
DETECTION_COOLDOWN = 2.0
refimg_save_for_next = None  # Backup nach 3 Würfen




def mouse_callback(event, x, y, flags, param):
    global current_index, calib_img
    if event == cv2.EVENT_LBUTTONDOWN and current_index < len(POINT_NAMES):
        name = POINT_NAMES[current_index]
        clicked_points[name] = {"x": x, "y": y}

        cv2.circle(calib_img, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(calib_img, name, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        current_index += 1
        cv2.imshow("KALIBRIERUNG - Dartscheibe (ESC/Q Ende)", calib_img)

        if current_index == len(POINT_NAMES):
            print("Alle Punkte gesetzt – ESC oder Q zum Beenden")

def calibrate_dartboard():
    global calib_img, calib_window_active, clicked_points, current_index
    
    try:
        frame = capture_frame()
        calib_img = frame.copy()
        num = f"{game_round_counter:03d}"
#        cv2.imwrite(f"calib_img_ref_{num}.jpg", calib_img)
        # In calibrate_dartboard(), Zeile mit cv2.imwrite:
        cv2.imwrite(os.path.join("current_game", f"calib_img_ref_{num}.jpg"), calib_img)

            
        clicked_points = {}
        current_index = 0
        
        calib_window_active = True
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
            if cv2.getWindowProperty("KALIBRIERUNG - Dartscheibe (ESC/Q Ende)", cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cv2.destroyAllWindows()
        calib_window_active = False
        
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


# 🔥 Oben in deine Imports hinzufügen:
#from ultralytics import YOLO
YOLO_MODEL_BOARD_PATH = "/home/matthias/PRIV/Darts_ML_detect_board/runs/detect/runs/darts_gpu_board/train3/weights/best.pt"  # Dein trainiertes Modell!

def auto_calibrate_dartboard():
    global cap, JSON_PATH, game_round_counter
    
    try:
        # 1. Foto von Webcam
        print("📸 Webcam-Foto...")
        frame = capture_frame()
        num = f"{game_round_counter:03d}"
        calib_img_path = os.path.join("current_game", f"auto_calib_img_{num}.jpg")
        cv2.imwrite(calib_img_path, frame)
        print(f"💾 {calib_img_path}")
        
        # 🔥 2. YOLO KALIBRIERUNG (ersetzt HoughCircles!)
        print("🤖 YOLO erkennt Randpunkte...")
        
        # Modell laden (einmalig)
        model = YOLO(YOLO_MODEL_BOARD_PATH)
        
        # YOLO Prediction (conf=0.15 basierend auf mAP=0.95!)
        results = model(frame, conf=0.5, verbose=False, imgsz=1024)
        result = results[0]
        
        # Punkt-Mapping (deine Reihenfolge!)
        point_names = ["P_05_20", "P_13_06", "P_17_03", "P_08_11"]
        clicked_points = {}
        
        if result.boxes is not None and len(result.boxes) > 0:
            # Nach Klasse sortieren (0=P_05_20, 1=P_13_06, ...)
            detections = []
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                detections.append({
                    'class': cls,
                    'conf': conf,
                    'x': center_x,
                    'y': center_y
                })
            
            # Sortiere nach Klasse (0,1,2,3)
            detections.sort(key=lambda d: d['class'])
            
            # Erstelle clicked_points
            for i, detection in enumerate(detections[:4]):  # Max 4 Punkte
                if i < len(point_names):
                    name = point_names[i]
                    clicked_points[name] = {
                        "x": detection['x'], 
                        "y": detection['y'],
                        "conf": detection['conf']
                    }
                    print(f"✅ {name}: ({detection['x']}, {detection['y']}) conf={detection['conf']:.2f}")
        
        # Fallback: Wenn YOLO <4 Punkte findet
        if len(clicked_points) < 4:
            print(f"⚠️ YOLO fand nur {len(clicked_points)}/4 Punkte → Manuelle Kalibrierung empfohlen!")
            messagebox.showwarning("YOLO", f"Nur {len(clicked_points)}/4 Punkte erkannt!\n→ Manuell kalibrieren?")
            return False
        
        # 4. Visualisierung
        viz_img = frame.copy()
        
        # YOLO Bounding Boxes zeichnen
        for name, pt in clicked_points.items():
            x, y = pt["x"], pt["y"]
            conf = pt.get("conf", 0)
            
            # Farbe nach Klasse
            colors = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 165, 0)]
            color_idx = point_names.index(name)
            color = colors[color_idx]
            
            # Großer Punkt + Confidence
            cv2.circle(viz_img, (x, y), 15, color, -1)
            cv2.circle(viz_img, (x, y), 20, color, 3)
            cv2.putText(viz_img, f"{name} ({conf:.1f})", (x+20, y-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        viz_path = os.path.join("current_game", f"yolo_calib_viz_{num}.jpg")
        cv2.imwrite(viz_path, viz_img)
        
        # 5. JSON speichern (kompatibel mit load_calibration()!)
        data = {
            "image": f"auto_calib_img_{num}.jpg",
            "refimage": f"ref_empty_throw_{num}.jpg",
            "points": clicked_points,
            "auto_detected": True,
            "yolo_detected": True,
            "detections": len(clicked_points)
        }
        
        with open(JSON_PATH, "w") as f:
            json.dump(data, f, indent=4)
        
        print(f"✅ YOLO-KALIBRIERUNG FERTIG: {JSON_PATH}")
        messagebox.showinfo("✅ YOLO Erfolg!", 
                           f"{len(clicked_points)}/4 Punkte erkannt!\n📁 {viz_path}")
        
        # Zeige Ergebnis
        cv2.imshow("✅ YOLO AUTO-KALIBRIERUNG", viz_img)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
        
        load_calibration()

        return True
        
    except Exception as e:
        print(f"❌ YOLO Fehler: {e}")
        messagebox.showerror("YOLO Fehler", f"YOLO-Kalibrierung fehlgeschlagen:\n{str(e)}")
        return False



# **AUTO MODE TOGGLE** (neuer Knopf)
def toggle_auto_mode():
    global auto_mode_active
    auto_mode_active = not auto_mode_active
    
    if auto_mode_active:
        btn_auto_toggle.config(text="Auto mode: AN", bg="lightpink")
        print("🔥 AUTO-DETEKT AKTIV!")
    else:
        btn_auto_toggle.config(text="Auto mode: AUS", bg="lightpink")
        print("⏹️ AUTO-DETEKT DEAKTIV!")


# -----------------------------
# **NEUE NUMMERIERUNG FUNKTION**
# -----------------------------
def get_next_filename(img_type="img"):
    global ref_group_counter, img_counter, throws_in_group
    
    game_round = f"{game_round_counter:03d}"
    cur_player= f"{current_player:02d}"
    
    if img_type == "ref":
        filename = f"ref_empty_round_{game_round}_player_{cur_player}.jpg"
        img_counter = 0
        throws_in_group = 0
        print(f"📸 REFERENZ gespeichert: {filename} (Runde {game_round})")
        return filename
    
    elif img_type == "img":
        throws_in_group += 1
        img_counter += 1
        
        if throws_in_group >= 3:
            # Gruppe komplett -> nächste Referenz-Gruppe
            ref_group_counter += 1
            img_counter = 0
            throws_in_group = 0
            print(f"🎉 Runde {game_round} FERTIG (3 Würfe)! -> Nächster Spieler: {ref_group_counter:03d}")
        
        throw_names = ["1st_throw", "2nd_throw", "3rd_throw"]
        throw_name = throw_names[throws_in_group - 1]
        filename = f"img_round_{game_round}_player_{cur_player}_{throw_name}.jpg" 
        # print(f"🎯 DART gespeichert: {filename} (Runde {game_round}, Wurf {throws_in_group}/3)")
        return filename


def save_numbered_images_with_reference(img, detected_point=None, current_player=None, score=None,  diff=None, diff_thresh=None):
    global refimg, refimg_filename
    
    def get_timestamp():
        return time.strftime("%Y%m%d%H%M%S")
    
    ts = get_timestamp()  # 🔥 NEU

    # **CURRENT_GAME ORGNER erstellen**
    game_folder = "current_game"
    if not os.path.exists(game_folder):
        os.makedirs(game_folder)
    # game_round_counter
    # **1. ZUERST Dateiname generieren (img_1st_throw_001.jpg)**
    img_filename = get_next_filename("img")
    
    # **VOLLSTÄNDIGER Dateiname für diff/debug (OHNE .jpg!)**
    base_filename = img_filename.replace('.jpg', '')  # "img_1st_throw_001"
    base_filename = base_filename+f"_{ts}"
    
    cv2.imwrite(os.path.join(game_folder, f"{base_filename}.jpg"), img)
    
    
    # **2. Diff-Bilder mit GLEICHEM Namen wie img_*** IM ORGNER**
    if diff is not None:
        diff_filename = f"diff_{base_filename}.jpg"  # diff_img_1st_throw_001.jpg
        cv2.imwrite(os.path.join(game_folder, diff_filename), diff)
    
    if diff_thresh is not None:
        diff_thresh_filename = f"diff_thresh_{base_filename}.jpg"  # diff_thresh_img_1st_throw_001.jpg
        cv2.imwrite(os.path.join(game_folder, diff_thresh_filename), diff_thresh)
    
    # **3. DEBUG mit Spieler + SCORE RASTER IM ORGNER**
    debug_orig = img.copy()
    if detected_point:
        
        # 🔥 NEU: DART PUNKT (grün/groß + rot/klein)
        cv2.circle(debug_orig, (int(detected_point[0]), int(detected_point[1])), 10, (0, 255, 0), 3)
        cv2.circle(debug_orig, (int(detected_point[0]), int(detected_point[1])), 3, (0, 0, 255), -1)
        
    # 🔥 NEU: SCORE RASTER zeichnen (über Homographie)
    if H is not None:
        # 20 Segmente + Ringe zeichnen
        draw_score_grid_on_image(debug_orig, H, img.shape[:2])
    
    # Spieler-Label
    player_color = (255, 0, 0) if current_player == 0 else (0, 0, 255)
    player_text = f"SPIELER {current_player+1}, Score: {score}"
    cv2.rectangle(debug_orig, (10, 10), (500, 50), player_color, -1)
    cv2.putText(debug_orig, player_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    debug_filename = f"debug_dart_{base_filename}.jpg"
    cv2.imwrite(os.path.join(game_folder, debug_filename), debug_orig)
    # print(f"🎯 {game_folder}/{debug_filename} + SCORE RASTER!")
    
    # print(f"✅ ALLES in {game_folder}/ gespeichert!")
    return base_filename

def draw_score_grid_on_image(img, H, img_shape):
    """20 Strahlen + Bulls/Triple/Double INNEN/ÄUSSEN PERSPEKTIVISCH"""
    h, w = img_shape[:2]
    
    try:
        inv_H = np.linalg.inv(H)
    except:
        print("❌ Homographie nicht invertierbar!")
        return
    
    angle_step = 2 * math.pi / 20
    start_angle = -math.pi / 2
    
    # **1. 20 STRAHLEN**
    for i in range(20):
        angle = start_angle + i * angle_step
        
        outer_bulls_warped = np.array([[[CENTER + R_OUTER_BULL * np.cos(angle), CENTER + R_OUTER_BULL * np.sin(angle)]]], np.float32)
        outer_bulls_orig = cv2.perspectiveTransform(outer_bulls_warped, inv_H)[0, 0]
        
        outer_warped = np.array([[[CENTER + RADIUS * np.cos(angle), CENTER + RADIUS * np.sin(angle)]]], np.float32)
        outer_orig = cv2.perspectiveTransform(outer_warped, inv_H)[0, 0]
        cv2.line(img, (int(outer_bulls_orig[0]), int(outer_bulls_orig[1])), (int(outer_orig[0]), int(outer_orig[1])), (255, 255, 255), 2)
        
        score_num = SEGMENTS[i]
        mid_pt_warped = np.array([[[CENTER + 0.7*RADIUS * np.cos(angle+(angle_step/2)), CENTER + 0.7*RADIUS * np.sin(angle+(angle_step/2))]]], np.float32)
        mid_pt_orig = cv2.perspectiveTransform(mid_pt_warped, inv_H)[0, 0]
        cv2.putText(img, str(score_num), (int(mid_pt_orig[0]-15), int(mid_pt_orig[1]+5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # **2. HILFSFUNKTION für PERSPEKTIVISCHE KREISE**
    def draw_perspective_ring(img, inv_H, radius_inner, radius_outer, color_inner, color_outer, thickness=3):
        """Innere + äußere Ringbegrenzung"""
        for radius, color in [(radius_inner, color_inner), (radius_outer, color_outer)]:
            points = []
            for i in range(36):  # 36 Punkte = glatter Kreis
                angle = i * 10 * math.pi / 180
                pt_warped = np.array([[[CENTER + radius * math.cos(angle),
                                     CENTER + radius * math.sin(angle)]]], np.float32)
                pt_orig = cv2.perspectiveTransform(pt_warped, inv_H)[0, 0]
                points.append(pt_orig.astype(int))
            cv2.polylines(img, [np.array(points)], True, color, thickness)
    
    # **BULLSEYE - INNER (50) + ÄUSSER (25)**
    draw_perspective_ring(img, inv_H, R_INNER_BULL, R_OUTER_BULL, 
                         (0, 255, 0), (0, 255, 128), 2)  # Hellgrün + Dunkelgrün
    
    # **TRIPLE RING Runde 001 FERTIG (3 Würfe)- INNER + ÄUSSER**
    draw_perspective_ring(img, inv_H, R_TRIPLE_INNER, R_TRIPLE_OUTER, 
                         (200, 0, 0), (255, 0, 0), 2)  # Lila + Dunkelblau
    
    # **DOUBLE RING - INNER + ÄUSSER**
    draw_perspective_ring(img, inv_H, R_DOUBLE_INNER, R_DOUBLE_OUTER, 
                         (0, 10, 255), (0, 0, 255), 2)  # Hellblau + Rot
    
    # print("✅ 20 Strahlen + Bulls/Triple/Double INNER/ÄUSSER!")



# -----------------------------
# Rest der Funktionen (unverändert oder leicht angepasst)
# -----------------------------
def init_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(CAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, ORIG_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ORIG_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
    return cap.isOpened()


def draw_cross(img, center, size=15, color=(0, 255, 0), thickness=2):
    """Zeichnet Kreuz statt Kreis"""
    x, y = center
    # Horizontale Linie
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    # Vertikale Linie  
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)


# 🔥 **AUTO-TRIGGER IM STREAM** (ersetzt update_stream())
def update_stream():
    global stream_running, auto_mode_active, last_detection_time, refimg, H, clicked_points
    
    if stream_running and cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_display = cv2.resize(frame, (854, 480))  # Für Tkinter
            
            # 🔥 **KALIBRIERUNGSPUNKTE ANZEIGEN** (wenn Kalibrierung geladen)
            if H is not None and os.path.exists(JSON_PATH):
                try:
                    # JSON Punkte laden
                    with open(JSON_PATH, "r") as f:
                        data = json.load(f)
                    pts = data["points"]
                    
                    # Punkte zeichnen (scale für Stream-Overlay)
                    scale_factor = frame_display.shape[1] / 1920  # 854/1920 ≈ 0.44
                    
                    for i, name in enumerate(POINT_NAMES[1:]):  # Ohne "Mitte"
                        if name in pts:
                            x = int(pts[name]["x"] * scale_factor)
                            y = int(pts[name]["y"] * scale_factor)
                            
                            # Farbige Punkte + Labels
                            colors = [(0, 255, 255), (255, 255, 255), (0,255,0), (255,125,0)]
                            color = colors[i % len(colors)]
                            
                            draw_cross(frame_display, (x, y), size=8, color=color, thickness=2)
                            cv2.circle(frame_display, (x, y), 12, color, 2)
                            cv2.putText(frame_display, name[:4], (x+15, y-5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                except:
                    pass  # JSON Fehler ignorieren
            
            # **AUTO-DETECTION** (bestehende Logik)
            if auto_mode_active and refimg is not None:
                try:
                    h, w = frame.shape[:2]
                    ref_resized = cv2.resize(refimg, (w, h))
                    diff = cv2.absdiff(frame, ref_resized)
                    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    _, diff_thresh = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
                    
                    change_pixels = np.sum(diff_thresh == 255)
                    current_time = time.time()
                    
                    if change_pixels > 8000 and (current_time - last_detection_time) > 1.5:
                        print(f"🎯 DART STECKT! Pixel: {change_pixels} → BUTTON AUTO-TRIGGER!")
                        last_detection_time = current_time
                        on_analyze_one_dart()
                    
                    # # DEBUG INFO
                    # cv2.rectangle(frame_display, (10, 45), (250, 75), (0, 0, 0), -1)
                    # cv2.putText(frame_display, f"AUTO: {int(change_pixels/100)}", (15, 65), 
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                except:
                    pass
            
            # **AUTO ON Overlay**
            if auto_mode_active:
                cv2.rectangle(frame_display, (0, 450), (150, 450), (0, 0, 255), -1)
                cv2.putText(frame_display, "AUTO Detection ON", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else :
                cv2.rectangle(frame_display, (0, 450), (150, 450), (0, 0, 255), -1)
                cv2.putText(frame_display, "AUTO Detection OFF", (0, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # **Stream anzeigen**
            frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
        
        video_label.after(50, update_stream)  # 20 FPS

def start_stream():
    global stream_running
    if init_camera():
        stream_running = True
        update_stream()
        btn_stream.config(text="Stream STOP", bg="gainsboro")
    else:
        messagebox.showerror("Fehler", f"Kamera /dev/video{CAM_INDEX} nicht verfügbar!")

def stop_stream():
    global stream_running
    stream_running = False
    btn_stream.config(text="Stream START", bg="lightgreen")

def load_calibration():
    global H
    if not os.path.exists(JSON_PATH):
        messagebox.showerror("Fehler", f"JSON nicht gefunden: {JSON_PATH}")
        return False
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    pts = data["points"]
    src_pts = np.array([[pts[name]["x"] * scale_x, pts[name]["y"] * scale_y] for name in POINT_NAMES[1:]], dtype=np.float32)
    dst_pts = np.array([[CENTER, CENTER - RADIUS], [CENTER + RADIUS, CENTER], [CENTER, CENTER + RADIUS], [CENTER - RADIUS, CENTER]], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    print("✅ Homographie Matrix geladen")
    return True

def capture_frame():
    global cap
    if not cap or not cap.isOpened():
        init_camera()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Konnte kein Bild von der Kamera lesen")
    return cv2.resize(frame, (ORIG_W, ORIG_H))

def compute_score_from_tip(exact_tip):
    if exact_tip is None:
        return 0
    angle_step = 2 * math.pi / 20
    start_angle = -math.pi / 2
    r_tip = math.hypot(exact_tip[0] - CENTER, exact_tip[1] - CENTER)
    angle_tip = math.atan2(exact_tip[1] - CENTER, exact_tip[0] - CENTER) - start_angle
    if angle_tip < 0: angle_tip += 2 * math.pi
    idx = int(angle_tip / angle_step) % 20
    score = SEGMENTS[idx]
    if r_tip > R_DOUBLE_OUTER : return 0 , "Mis" 
    elif r_tip > R_DOUBLE_INNER and r_tip < R_DOUBLE_OUTER : return score * 2 , f"D{score:>2}" 
    elif r_tip > R_TRIPLE_INNER and r_tip < R_TRIPLE_OUTER : return score * 3 , f"T{score:>2}" 
    elif r_tip < R_OUTER_BULL and r_tip > R_INNER_BULL: return 25 , "B25" 
    elif r_tip < R_INNER_BULL: return 50 ,  "B50"
    else: return score, f"S{score:>2}" 
    
    
def on_capture_reference():
    global refimg, refimg_filename, dart_counter_in_group, round_points, throws_label, points_detail_label, ref_group_counter, auto_mode_active
    
    try:
        frame = capture_frame()
        refimg_filename = get_next_filename("ref")
        refimg = frame.copy()
        cv2.imwrite(os.path.join("current_game", refimg_filename), refimg)
        ref_group_counter += 1
        dart_counter_in_group = 0
        round_points = []
        throws_label.config(text="0/3")
        points_detail_label.config(text="PTS: ---", bg="lightgray")
        status_label.config(text=f"Referenz aufgenommen, {player_names[current_player]} kann werfen, Runde {game_round_counter:03d}", bg="lightgreen")
        lbl_current_var.set(f"Am Pfeil: {player_names[current_player]}")

        # 🔥 **AUTO-MODE AUTOMATISCH STARTEN!**
        if not auto_mode_active:
            auto_mode_active = True
            btn_auto_toggle.config(text="AUTO AUS", bg="grey")
            print("🚀 AUTO-MODE AUTO-START nach Referenz!")
            
    except Exception as e:
        status_label.config(text=f"❌ Referenz-Fehler: {str(e)}", bg="red")

def on_analyze_one_dart():
    global refimg, H, player_scores, current_player, dart_counter_in_group, round_points, str_round_points, auto_mode_active
 
    
 
    def get_best_dart_tip(results):
        """
        🔥 Extrahiert beste Dartspitze aus YOLO results
        
        Args:
            results: YOLO results object
        
        Returns:
            tuple: (best_tip, center_x, center_y) oder (None, 0, 0)
        """
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
                    return best_tip, center_x, center_y  # Erste Detection reicht!
        
        print("❌ Kein Dart erkannt")
        return None, 0, 0

    
    def get_score_from_tip(best_tip, H):
        """
        🔥 Berechnet Score aus Dartspitze
        
        Args:
            best_tip: (x, y) Koordinaten oder None
            H: Homographie-Matrix
        
        Returns:
            tuple: (points, str_points, points_color)
        """
        if best_tip is None:
            return 0, " 0",
        
        try:
            # Perspective Transform
            point_array = np.array([[[float(best_tip[0]), float(best_tip[1])]]], dtype=np.float32)
            warped = cv2.perspectiveTransform(point_array, H)
            exact_tip = (int(warped[0, 0, 0]), int(warped[0, 0, 1]))
            
            # Score berechnen
            points, str_points = compute_score_from_tip(exact_tip)
            print(f"📍 Tip: {best_tip} → {exact_tip} → {points} {str_points}")
            
            return points, str_points
            
        except Exception as e:
            print(f"❌ Transform Fehler: {e}")
            return 0, "ERR"

    
    if refimg is None: 
        status_label.config(text="❌ Zuerst REFERENZ aufnehmen!", bg="red")
        root.after(2000, lambda: status_label.config(text="Darts-System | REFERENZ + 3 Würfe pro Runde", bg="lightgreen"))
        return
    
    # 🔥 KRITISCHER FIX: H laden!
    if H is None:
        if not load_calibration():
            status_label.config(text="❌ Zuerst KALIBRIEREN!", bg="red")
            return
    
    try:
        img = capture_frame()
        if img is None:
            status_label.config(text="❌ Kamera-Fehler!", bg="red")
            return
        print(" on_analyze_one_dart try 1 ")
            
        # Rest unverändert...
        dart_counter_in_group += 1
        throws_label.config(text=f"{dart_counter_in_group}/3")
        
        h, w = img.shape[:2]
        refimg_resized = cv2.resize(refimg, (w, h))
        diff = cv2.absdiff(img, refimg_resized)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff_thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        
        model = YOLO(YOLO_WEIGHTS)
        # results_img = model(refimg_resized, device="cpu", conf=0.1, verbose=False)
        results = model(diff, device="cpu", conf=0.3, verbose=False)
        # diff_rgb = cv2.cvtColor(diff_thresh, cv2.COLOR_GRAY2RGB)
        
        best_tip, center_x, center_y = get_best_dart_tip(results)
        detected_point = [center_x, center_y] if best_tip else None
        points, str_points = get_score_from_tip(best_tip, H)

        
        save_numbered_images_with_reference(
            img, detected_point=detected_point, current_player=current_player, 
            score= str_points, 
            diff=diff, diff_thresh=diff_thresh
        )
        
        round_points.append(points)
        str_round_points.append(str_points)
        total_round = sum(round_points)
        if dart_counter_in_group == 1:
            points_text = f"PTS: {str_points} => {total_round}"
        else:
            points_text = f"PTS: {str_round_points[0]} + {str_points} => {total_round}"
        
        points_detail_label.config(text=points_text, bg="lightgreen")
        
        old_score = player_scores[current_player]
        new_score = max(0, old_score - points)
        player_scores[current_player] = new_score
        lbl_p1_var.set(f"{player_names[0]}: {player_scores[0]}")
        lbl_p2_var.set(f"{player_names[1]}: {player_scores[1]}")
        
        # update refimage:                
        refimg = img

        if dart_counter_in_group >= 3:
            # **1. Scores aktualisieren**
            next_player = 1 - current_player
            lbl_current_var.set(f"Nächster Spieler: {player_names[next_player]}")
            
            points_list_text = " + ".join(map(str, str_round_points)) + f" => {sum(round_points)}"
            points_detail_label.config(text=f"Beendet: {points_list_text} PTS!", bg="gold")
            throws_label.config(text="3/3")
            status_label.config(text=f"{player_names[current_player]} hat geworfen! Pfeile raus. Dann ist {player_names[next_player]} dran. Aber erst neue Referenz aufnehmen! ...", bg="lightblue")
            
            # 🔥 **2. AUTO-MODE OFF + refimg BEWACHTEN (WICHTIG!)**
            global refimg_save_for_next  # Backup für nächsten Spieler
            refimg_save_for_next = refimg.copy()  # Letztes Bild mit 3 Darts sichern
            refimg = None  # ← Diff-Logik stoppen!
            
            auto_mode_active = False
            # btn_auto_toggle.config(text="🚀 AUTO ON", bg="lightgrey")
            print("AUTO-MODE AUTO-OFF nach 3 Würfen!")
            
            if current_player == (len(player_names) - 1):
                global game_round_counter, counter_label
                print("   NEUE RUNDE  ########################################################################################## " )
                game_round_counter += 1
                counter_label.config(text=f"{game_round_counter:03d}")
                status_label.config(text=f"RUNDE BEENDET! NEUE REFERENZ AUFNEHMEN! ... {player_names[next_player]} ist dran" , bg="gold")
            
            current_player = next_player
            round_points = []
            str_round_points = []
    
                
    except Exception as e:
        print(" on_analyze_one_dart EXEPTION ")

        status_label.config(text=f"ANALYSE-FEHLER: {str(e)}", bg="red")


        
def reset_round():
    global dart_counter_in_group, round_points, throws_label, points_label
    dart_counter_in_group = 0
    round_points = []
    throws_label.config(text="0/3")
    points_label.config(text="0", bg="lightgray")
        

def on_close():
    global cap, stream_running, cv2_module
    stream_running = False
    if cap: cap.release()
    if 'cv2_module' in globals() and cv2_module:
        cv2_module.destroyAllWindows()
    root.destroy()


root = tk.Tk()
root.title("Darts Scoring - AUTO-DETECTION 🔥")
root.geometry("1400x850")

status_label = tk.Label(root, text="Darts-System | REFERENZ + AUTO-DETECTION", font=("Arial", 14), bg="lightgreen")
status_label.pack(pady=10)

# **STATUSZEILE erweitert**
status_counter_frame = tk.Frame(root, bg='lightgray')
status_counter_frame.pack(fill=tk.X, padx=10, pady=5)

tk.Label(status_counter_frame, text="Runde:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
counter_label = tk.Label(status_counter_frame, text=f"{game_round_counter:03d}", font=("Arial", 16, "bold"), bg="yellow", width=6)
counter_label.pack(side=tk.LEFT, padx=5)

tk.Label(status_counter_frame, text="| Würfe:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)
throws_label = tk.Label(status_counter_frame, text="0/3", font=("Arial", 16, "bold"), bg="lightblue", width=6)
throws_label.pack(side=tk.LEFT, padx=5)

tk.Label(status_counter_frame, text="|", font=("Arial", 12)).pack(side=tk.LEFT)
points_detail_label = tk.Label(status_counter_frame, text=f"{player_names[current_player]}, STARTE SPIEL", font=("Arial", 14, "bold"), bg="lightgreen", width=30)
points_detail_label.pack(side=tk.LEFT, padx=5)

# **🔥 NEU: AUTO STATUS**
tk.Label(status_counter_frame, text="| AUTO:", font=("Arial", 12, "bold")).pack(side=tk.LEFT)

# Video + Controls (unverändert)
video_frame = tk.Frame(root, bg='gray', width=1000, height=600)
video_frame.pack(side=tk.LEFT, padx=10, pady=10)
video_frame.pack_propagate(False)
video_label = tk.Label(video_frame, bg="black", text="Kamera-Bild erscheint hier")
video_label.pack(expand=True)

control_frame = tk.Frame(root, bg='lightgray', width=350)
control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
control_frame.pack_propagate(False)

# Buttons (AUTO + bestehende)
btn_calib = tk.Button(control_frame, text="Manuell kalibrieren", command=calibrate_dartboard, bg="gainsboro", fg="white", font=("Arial", 11, "bold"), height=2)
btn_calib.pack(fill=tk.X, pady=3)

btn_auto_calib = tk.Button(control_frame, text="Automatisch kalibrieren", command=auto_calibrate_dartboard, bg="lightgreen", fg="white", font=("Arial", 11, "bold"), height=2)
btn_auto_calib.pack(fill=tk.X, pady=3)

btn_stream = tk.Button(control_frame, text="Stream START", command=lambda: start_stream() if not stream_running else stop_stream(), bg="lightgreen", font=("Arial", 12, "bold"), height=2)
btn_stream.pack(fill=tk.X, pady=5)


btn_auto_toggle = tk.Button(control_frame, text="toggle automatic detection mode", 
                    command=lambda: toggle_auto_mode(),
                    bg="gainsboro", fg="white", font=("Arial", 11), height=2)
btn_auto_toggle.pack(fill=tk.X, pady=5)

btn_ref = tk.Button(control_frame, text="Referenzbild aufnehmen ", command=on_capture_reference, bg="lightcoral", font=("Arial", 14, "bold"), height=4)
btn_ref.pack(fill=tk.X, pady=5)

# **MANUAL bleibt als Fallback**
btn_dart = tk.Button(control_frame, text="Detektiere Pfeil manuell", command=on_analyze_one_dart, bg="lightcoral", font=("Arial", 14, "bold"), height=4)
btn_dart.pack(fill=tk.X, pady=5)

# Scores (unverändert)
score_frame = tk.LabelFrame(control_frame, text="Scores (301)", font=("Arial", 12))
score_frame.pack(fill=tk.X, pady=10)

lbl_p1_var = tk.StringVar(value=f"{player_names[0]}: {player_scores[0]}")
lbl_p2_var = tk.StringVar(value=f"{player_names[1]}: {player_scores[1]}")
lbl_current_var = tk.StringVar(value=f"Am Pfeil: {player_names[current_player]}")

tk.Label(score_frame, textvariable=lbl_p1_var, font=("Arial", 20, "bold")).pack(pady=10)
tk.Label(score_frame, textvariable=lbl_p2_var, font=("Arial", 20, "bold")).pack(pady=10)
tk.Label(score_frame, textvariable=lbl_current_var, font=("Arial", 16, "bold")).pack(pady=10)

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()