# game/logic.py
from typing import List
from .models import Player

class Game:
    def __init__(self, players: List[str], start_score: int = 501):
        self.players: List[Player] = [Player(name=p, score=start_score) for p in players]
        self.current_index: int = 0
        self.finished: bool = False
        self.winner: Player | None = None

    @property
    def current_player(self) -> Player:
        return self.players[self.current_index]

    def register_throw(self, darts: List[int]) -> None:
        if self.finished:
            return
        player = self.current_player
        old_score = player.score
        player.apply_throw(darts)
        if player.score == 0:
            self.finished = True
            self.winner = player
            return
        # zb einfache „nächster Spieler“-Logik
        if old_score == player.score:  # Bust o.ä.
            pass  # hier kannst du deine Regeln anpassen
        self.current_index = (self.current_index + 1) % len(self.players)
