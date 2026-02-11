import os
import time
import cv2
from pathlib import Path
from typing import Optional, Tuple, Any

# NUR echte Konstanten importieren
from src.config.constants import ORIG_W

def get_next_filename(img_type: str = "img", game_round: int = 1, player_idx: int = 0) -> str:
    """Generiert nummerierte Dateinamen für Referenz/Würfe"""
    game_round_str = f"{game_round:03d}"
    cur_player_str = f"{player_idx:02d}"
    
    if img_type == "ref":
        return f"ref_empty_round_{game_round_str}_player_{cur_player_str}.jpg"
    
    # Für Würfe: 1st_throw, 2nd_throw, 3rd_throw
    throw_names = ["1st_throw", "2nd_throw", "3rd_throw"]
    # throws_in_group muss von aufrufender Funktion kommen
    throw_idx = 1  # Default, wird von main.py übergeben
    throw_name = throw_names[throw_idx - 1] if throw_idx <= 3 else "extra_throw"
    
    return f"img_round_{game_round_str}_player_{cur_player_str}_{throw_name}.jpg"

def save_numbered_images_with_reference(
    img: Any, 
    detected_point: Optional[Tuple[int, int]] = None,
    current_player: Optional[int] = None, 
    score: Optional[str] = None,
    diff: Optional[Any] = None, 
    diff_thresh: Optional[Any] = None,
    game_round: int = 1  # Neu: Parameter hinzufügen
) -> str:
    """Speichert Bildgruppe mit Debug-Info"""
    game_folder = "current_game"
    os.makedirs(game_folder, exist_ok=True)
    
    ts = time.strftime("%Y%m%d%H%M%S")
    img_filename = get_next_filename("img", game_round, current_player)
    base_filename = img_filename.replace('.jpg', f"_{ts}")
    
    # Haupbild
    cv2.imwrite(os.path.join(game_folder, f"{base_filename}.jpg"), img)
    
    # Diff-Bilder
    if diff is not None:
        cv2.imwrite(os.path.join(game_folder, f"diff_{base_filename}.jpg"), diff)
    if diff_thresh is not None:
        cv2.imwrite(os.path.join(game_folder, f"diff_thresh_{base_filename}.jpg"), diff_thresh)
    
    # Debug-Bild mit Overlay
    debug_img = img.copy()
    if detected_point:
        cv2.circle(debug_img, (int(detected_point[0]), int(detected_point[1])), 10, (0, 255, 0), 3)
        cv2.circle(debug_img, (int(detected_point[0]), int(detected_point[1])), 3, (0, 0, 255), -1)
    
    # Spieler-Label
    player_color = (255, 0, 0) if current_player == 0 else (0, 0, 255)
    player_text = f"SPIELER {current_player+1}, Score: {score}"
    cv2.rectangle(debug_img, (10, 10), (500, 50), player_color, -1)
    cv2.putText(debug_img, player_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    debug_filename = f"debug_dart_{base_filename}.jpg"
    cv2.imwrite(os.path.join(game_folder, debug_filename), debug_img)
    
    print(f"✅ ALLES in {game_folder}/ gespeichert!")
    return base_filename
