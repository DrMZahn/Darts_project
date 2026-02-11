import cv2
from typing import Optional
from src.config.constants import ORIG_W, ORIG_H, CAM_INDEX

class WebcamManager:
    def __init__(self, cam_index: int = CAM_INDEX):
        self.cap = None
        self.cam_index = cam_index
    
    def init_camera(self) -> bool:
        """Initialisiert Kamera"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.cam_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ORIG_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ORIG_H)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        return self.cap.isOpened()
    
    def capture_frame(self) -> Optional[cv2.Mat]:
        """Erfasst und resized Frame"""
        if not self.cap or not self.cap.isOpened():
            self.init_camera()
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Konnte kein Bild von der Kamera lesen")
        return cv2.resize(frame, (ORIG_W, ORIG_H))
    
    def release(self):
        """Kamera freigeben"""
        if self.cap:
            self.cap.release()
