# ui/app.py
import tkinter as tk
from game.logic import Game
from vision.capture import capture_board_frame
from vision.nn_inference import DartsNetWrapper

class DartsApp:
    def __init__(self, root: tk.Tk, players: list[str]):
        self.root = root
        self.game = Game(players)
        self.net = DartsNetWrapper("pfad/zum/modell")
        self._build_ui()

    def _build_ui(self):
        self.root.title("Dart-Scorer")
        self.labels = {}
        row = 0
        for p in self.game.players:
            lbl_name = tk.Label(self.root, text=p.name)
            lbl_name.grid(row=row, column=0, padx=10, pady=5)
            lbl_score = tk.Label(self.root, text=str(p.score))
            lbl_score.grid(row=row, column=1, padx=10, pady=5)
            self.labels[p.name] = lbl_score
            row += 1

        self.status = tk.Label(self.root, text="Bereit")
        self.status.grid(row=row, column=0, columnspan=2, pady=10)

        self.btn_throw = tk.Button(self.root, text="3 Darts auswerten", command=self.on_throw)
        self.btn_throw.grid(row=row+1, column=0, columnspan=2, pady=10)

    def refresh_scores(self):
        for p in self.game.players:
            self.labels[p.name].config(text=str(p.score))
        if self.game.finished:
            self.status.config(text=f"Spielende! Gewinner: {self.game.winner.name}")
        else:
            self.status.config(text=f"Am Zug: {self.game.current_player.name}")

    def on_throw(self):
        if self.game.finished:
            return
        try:
            frame = capture_board_frame()
            scores = self.net.predict_scores(frame)
            if len(scores) != 3:
                self.status.config(text="Fehler: NN lieferte nicht genau 3 Werte")
                return
            self.game.register_throw(scores)
            self.refresh_scores()
        except Exception as e:
            self.status.config(text=f"Fehler: {e}")

def main():
    root = tk.Tk()
    app = DartsApp(root, players=["Spieler 1", "Spieler 2"])
    app.refresh_scores()
    root.mainloop()

if __name__ == "__main__":
    main()
