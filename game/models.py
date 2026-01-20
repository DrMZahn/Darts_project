# game/models.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class ThrowResult:
    darts: List[int]  # z.B. [60, 45, 5]
    total: int

@dataclass
class Player:
    name: str
    score: int = 501
    history: List[ThrowResult] = field(default_factory=list)

    def apply_throw(self, darts: List[int]) -> None:
        total = sum(darts)
        new_score = self.score - total
        # Hier kannst du noch Double-Out-Regeln etc. einbauen
        if new_score >= 0:
            self.score = new_score
            self.history.append(ThrowResult(darts=darts, total=total))
        # sonst: Bust-Logik (Wurf zählt nicht)
