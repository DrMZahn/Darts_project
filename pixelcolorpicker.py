import cv2
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python pickel_farbe.py <bild.jpg>")
    sys.exit(1)

bild_pfad = sys.argv[1]

if not os.path.isfile(bild_pfad):
    print(f"Datei nicht gefunden: {bild_pfad}")
    sys.exit(1)

# Bild laden
bild = cv2.imread(bild_pfad)

if bild is None:
    print("Bild konnte nicht geladen werden")
    sys.exit(1)

def maus_klick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        b, g, r = bild[y, x]
        print(f"Position: ({x}, {y})")
        print(f"RGB: ({r}, {g}, {b})")
        print("-" * 30)

fenstername = f"Bild: {bild_pfad}"
cv2.imshow(fenstername, bild)
cv2.setMouseCallback(fenstername, maus_klick)

print("Linksklick = Farbe auslesen | ESC = Beenden")

while True:
    if cv2.waitKey(1) == 27:  # ESC
        break

cv2.destroyAllWindows()
