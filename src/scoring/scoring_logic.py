import math
import cv2
import numpy as np
from typing import Tuple, Dict
from src.config.constants import (
    CENTER, SEGMENTS, RADIUS, R_DOUBLE_INNER, R_DOUBLE_OUTER,
    R_TRIPLE_INNER, R_TRIPLE_OUTER, R_INNER_BULL, R_OUTER_BULL
)

def compute_score_from_tip(exact_tip: Tuple[float, float]) -> Tuple[int, str]:
    """DEINE ORIGINALE Funktion - unverändert!"""
    if exact_tip is None:
        return 0, "Mis" 
    
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


def is_double_checkout_valid(remaining_score: int, points: int, is_double: bool) -> bool:
    """Prüft ob Double Out gültig ist"""
    return points == remaining_score and is_double
