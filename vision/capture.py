import cv2

def capture_chicony():
    """Spezifisch für Chicony USB 2.0 Camera"""
    # Chicony: video0 oder video1 testen
    for cam_id in [0, 1]:  # Chicony video0/video1
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            # Chicony-Optimierung: Auflösung setzen
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame.shape[0] > 500:  # Gute Qualität
                print(f"✅ Chicony /dev/video{cam_id} OK")
                return cam_id
    
    raise RuntimeError("Chicony Camera nicht gefunden! (video0/video1)")

def capture_board_frame(camera_index=None):
    """Foto mit Chicony oder Fallback"""
    if camera_index is None:
        camera_index = capture_chicony()
    
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Chicony-optimiert
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise RuntimeError(f"Chicony /dev/video{camera_index} Fehler!")
    
    cv2.imwrite("latest_dart_chicony.jpg", frame)
    print(f"📸 Chicony Foto: {frame.shape}")
    return frame
