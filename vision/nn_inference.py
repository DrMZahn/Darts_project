# vision/nn_inference.py
# hier integrierst du dein eigenes Netz (PyTorch, TensorFlow o.ä.)
from typing import List
import numpy as np

# Pseudocode: du ersetzt dies durch dein echtes Modell
class DartsNetWrapper:
    def __init__(self, model_path: str):
        # Modell laden
        # self.model = ...
        pass

    def preprocess(self, frame) -> np.ndarray:
        # Resize, Normalisierung etc.
        return frame

    def predict_scores(self, frame) -> List[int]:
        """
        Erwartung: Liefert genau drei Punktwerte für die drei Darts.
        """
        x = self.preprocess(frame)
        # logits = self.model(x)
        # hier aus logits die Punktzahl bestimmen
        # Pseudocode:
        # scores = postprocess(logits)
        scores = [60, 45, 5]  # Dummy
        return scores
