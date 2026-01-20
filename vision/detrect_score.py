# import cv2
# import numpy as np
# import math

# def detect_dart_score(image_path):
#     """
#     Bestimmt den Score des Darts auf einem Dartsboard-Bild.
#     Input:  Pfad zum Bild
#     Output: geschätzter Score (int)
#     """
#     # Bild laden
#     img = cv2.imread(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (7, 7), 0)

#     # Kreis des Boards erkennen
#     circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
#                                param1=100, param2=30, minRadius=200, maxRadius=400)

#     if circles is None:
#         raise ValueError("Kein Board-Kreis erkannt.")

#     # Kreisparameter extrahieren
#     c = np.uint16(np.around(circles))[0][0]
#     board_center = (c[0], c[1])
#     board_radius = c[2]

#     # Kanten extrahieren – Dartspitze finden
#     edges = cv2.Canny(gray, 50, 150)
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     dart_tip = None
#     max_y = 0
#     for cnt in contours:
#         for p in cnt:
#             x, y = p[0]
#             if y > max_y:  # grobe Heuristik: tiefster Punkt = Dartspitze
#                 max_y = y
#                 dart_tip = (x, y)

#     if dart_tip is None:
#         raise ValueError("Kein Dart erkannt.")

#     # Winkel und Distanz zur Mitte
#     dx = dart_tip[0] - board_center[0]
#     dy = board_center[1] - dart_tip[1]  # um Koordinatensystem anzupassen
#     angle_deg = (math.degrees(math.atan2(dy, dx)) + 360) % 360
#     dist = math.sqrt(dx**2 + dy**2)

#     # Dartboard-Sektoren im Uhrzeigersinn ab Winkel 0° = 3 Uhr
#     sectors = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 
#                11, 8, 16, 7, 19, 3, 17, 2, 15, 10]

#     # Winkel pro Sektor = 18°
#     sector_index = int(((angle_deg + 9) % 360) / 18)
#     base_score = sectors[sector_index]

#     # Ring-Erkennung anhand der Distanz
#     r_norm = dist / board_radius
#     if r_norm < 0.07:
#         return 50                # Bullseye
#     elif r_norm < 0.13:
#         return 25                # Outer bull
#     elif 0.45 < r_norm < 0.5:
#         return base_score * 3     # Triple
#     elif 0.9 < r_norm < 1.0:
#         return base_score * 2     # Double
#     else:
#         return base_score         # Single


import cv2
import numpy as np
import math
from datetime import datetime
import os

def detect_dart_score_visual(image_path, show=True):
    """
    Erkennt Dart-Score + zeichnet ALLE 20 Felder der Dartsscheibe!
    Speichert Debug-Bild als images/test/darts_debug_YYYYMMDDHHMMSS.jpg
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Dartboard Kreis erkennen
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=100, param2=30, minRadius=200, maxRadius=400)
    if circles is None:
        raise ValueError("Kein Dartboard-Kreis erkannt.")
    
    c = np.uint16(np.around(circles))[0][0]
    center = (c[0], c[1])
    radius = c[2]

    # *** ALLE 20 DARTSFELDER zeichnen ***
    sectors = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 
               11, 8, 16, 7, 19, 3, 17, 2, 15, 10]
    
    # Mittellinie (vertikal)
    cv2.line(img, center, (center[0], center[1]-radius), (255,255,255), 3)
    
    # 20 Sektoren (je 18°)
    for i in range(20):
        angle1 = (i * 18) * math.pi / 180
        angle2 = ((i+1) * 18) * math.pi / 180
        
        # Eckpunkte des Sektors
        x1_1 = int(center[0] + radius * math.cos(angle1))
        y1_1 = int(center[1] - radius * math.sin(angle1))
        x1_2 = int(center[0] + radius * math.cos(angle2))
        y1_2 = int(center[1] - radius * math.sin(angle2))
        
        # Sektorlinien zeichnen
        cv2.line(img, center, (x1_1, y1_1), (0, 255, 0), 2)
        cv2.line(img, center, (x1_2, y1_2), (0, 255, 0), 2)
        
        # Feld-Nummer anzeigen
        mid_angle = (angle1 + angle2) / 2
        text_x = int(center[0] + (radius * 0.7) * math.cos(mid_angle))
        text_y = int(center[1] - (radius * 0.7) * math.sin(mid_angle))
        cv2.putText(img, str(sectors[i]), (text_x-10, text_y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # *** Double/Triple Ringe zeichnen ***
    double_radius = int(radius * 0.91)
    triple_radius = int(radius * 0.47)
    
    cv2.circle(img, center, double_radius, (0, 255, 255), 3)    # Double (Gelb)
    cv2.circle(img, center, triple_radius, (255, 0, 255), 3)   # Triple (Magenta)
    cv2.circle(img, center, int(radius*0.107), (255, 255, 0), 2)  # Outer Bull (Cyan)

    # Dartspitze finden
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dart_tip = None
    max_y = 0
    for cnt in contours:
        for p in cnt:
            x, y = p[0]
            if y > max_y:
                max_y = y
                dart_tip = (x, y)

    if dart_tip:
        # Visualisierung Dart
        cv2.circle(img, center, 5, (0, 255, 0), -1)      # Zentrum Grün
        cv2.circle(img, dart_tip, 8, (0, 0, 255), -1)    # Dart Rot
        cv2.line(img, center, dart_tip, (255, 0, 0), 3)  # Linie Blau

        # Score berechnen
        dx = dart_tip[0] - center[0]
        dy = center[1] - dart_tip[1]
        angle_deg = (math.degrees(math.atan2(dy, dx)) + 360) % 360
        dist = math.sqrt(dx**2 + dy**2)
        
        sector_index = int(((angle_deg + 9) % 360) / 18)
        base_score = sectors[sector_index]
        r_norm = dist / radius
        
        if r_norm < 0.07:
            score = 50; ring = "Bullseye"
        elif r_norm < 0.13:
            score = 25; ring = "Outer Bull"
        elif 0.45 < r_norm < 0.5:
            score = base_score * 3; ring = "Triple"
        elif 0.9 < r_norm < 1.0:
            score = base_score * 2; ring = "Double"
        else:
            score = base_score; ring = "Single"
    else:
        score = 0
        ring = "Kein Dart"

    # Score-Text
    cv2.putText(img, f"{ring}: {score}", (30, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # *** DEBUG-BILD SPEICHERN ***
    os.makedirs("images/test", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_filename = f"images/test/darts_debug_{timestamp}.jpg"
    cv2.imwrite(debug_filename, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"💾 DEBUG gespeichert: {debug_filename}")

    if show:
        cv2.imshow("Dartboard + ALLE FELDER", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return score


