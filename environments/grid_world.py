from typing import List
import numpy as np
import time


class ModelFreeEnv:
    """
    Contrat de base pour tous les environnements Model-Free.
    """
    def reset(self):
        raise NotImplementedError

    def step(self, action: int):
        raise NotImplementedError

    def is_game_over(self) -> bool:
        raise NotImplementedError

    def current_state(self) -> int:
        raise NotImplementedError

    def available_actions(self) -> List[int]:
        raise NotImplementedError

    def score(self) -> float:
        raise NotImplementedError

    def max_state_count(self) -> int:
        raise NotImplementedError

    def max_actions_count(self) -> int:
        raise NotImplementedError

    def pretty_print(self):
        raise NotImplementedError


class GridWorldEnv(ModelFreeEnv):
    """
    Environnement Grid World 5x5.

    Représentation :
        [A][ ][ ][ ][X]   ← état 4  = TRAP  (reward = -3.0, terminal)
        [ ][ ][ ][ ][ ]
        [ ][ ][ ][ ][ ]
        [ ][ ][ ][ ][ ]
        [ ][ ][ ][ ][G]   ← état 24 = GOAL  (reward = +1.0, terminal)

    Actions : 0=Haut, 1=Bas, 2=Gauche, 3=Droite
    Agent démarre en haut à gauche (état 0).
    """

    def __init__(self, rows: int = 5, cols: int = 5):
        self.rows = rows
        self.cols = cols
        self.trap = cols - 1               # état 4  → coin haut-droit
        self.goal = rows * cols - 1        # état 24 → coin bas-droit
        self.terminal_states = [self.trap, self.goal]
        self.agent_pos = 0
        self._score = 0.0                  # score cumulé

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _state_to_rc(self, state: int):
        """Convertit un état (int) en (ligne, colonne)."""
        return state // self.cols, state % self.cols

    def _rc_to_state(self, row: int, col: int) -> int:
        """Convertit (ligne, colonne) en état (int)."""
        return row * self.cols + col

    def _next_position(self, state: int, action: int) -> int:
        """Calcule la prochaine position sans modifier l'état."""
        row, col = self._state_to_rc(state)
        if action == 0:    # Haut
            row = max(row - 1, 0)
        elif action == 1:  # Bas
            row = min(row + 1, self.rows - 1)
        elif action == 2:  # Gauche
            col = max(col - 1, 0)
        elif action == 3:  # Droite
            col = min(col + 1, self.cols - 1)
        return self._rc_to_state(row, col)

    def _compute_reward(self, state: int) -> float:
        """Calcule le reward pour un état donné."""
        if state == self.goal:
            return 1.0
        elif state == self.trap:
            return -3.0
        return 0.0

    # ------------------------------------------------------------------
    # Contrat ModelFreeEnv
    # ------------------------------------------------------------------

    def reset(self):
        """Réinitialise l'environnement."""
        self.agent_pos = 0
        self._score = 0.0

    def step(self, action: int):
        """
        Joue une action. Ne retourne rien.
        Utilise current_state(), score(), is_game_over() pour les infos.
        """
        if action not in self.available_actions():
            raise Exception(f"Action invalide : {action}")
        if self.is_game_over():
            raise Exception("Le jeu est terminé, appelle reset() d'abord !")

        self.agent_pos = self._next_position(self.agent_pos, action)
        self._score += self._compute_reward(self.agent_pos)

    def is_game_over(self) -> bool:
        """Retourne True si l'agent est sur un état terminal."""
        return self.agent_pos in self.terminal_states

    def current_state(self) -> int:
        """Retourne la position actuelle de l'agent."""
        return self.agent_pos

    def available_actions(self) -> List[int]:
        """Retourne les actions disponibles : 0=Haut, 1=Bas, 2=Gauche, 3=Droite."""
        return [0, 1, 2, 3]

    def score(self) -> float:
        """Retourne le score cumulé."""
        return self._score

    def max_state_count(self) -> int:
        """Nombre total d'états possibles."""
        return self.rows * self.cols

    def max_actions_count(self) -> int:
        """Nombre total d'actions possibles."""
        return 4

    def pretty_print(self):
        """Affiche la grille dans le terminal."""
        ACTION_NAMES = ["↑", "↓", "←", "→"]
        print()
        for row in range(self.rows):
            line = ""
            for col in range(self.cols):
                s = self._rc_to_state(row, col)
                if s == self.agent_pos:
                    line += "[A]"
                elif s == self.goal:
                    line += "[G]"
                elif s == self.trap:
                    line += "[X]"
                else:
                    line += "[ ]"
            print(f"  {line}")
        print(f"  Score : {self._score:+.1f}")
        if self.is_game_over():
            if self.agent_pos == self.goal:
                print("  ✅ Gagné ! Reward = +1.0")
            else:
                print("  ❌ Perdu ! Reward = -3.0")

    # ------------------------------------------------------------------
    # Mode humain
    # ------------------------------------------------------------------

    def play_human(self):
        """Mode interactif : l'utilisateur joue manuellement."""
        print("\n🎮 MODE HUMAIN — Grid World")
        print("Commandes : [0]=Haut  [1]=Bas  [2]=Gauche  [3]=Droite  [q]=Quitter\n")

        self.reset()
        self.pretty_print()

        ACTION_NAMES = {0: "Haut", 1: "Bas", 2: "Gauche", 3: "Droite"}

        while not self.is_game_over():
            commande = input("  Ton action : ").strip().lower()

            if commande == "q":
                print("  Partie abandonnée.")
                break
            elif commande in ["0", "1", "2", "3"]:
                action = int(commande)
                prev_score = self.score()
                self.step(action)
                reward = self.score() - prev_score
                print(f"  → {ACTION_NAMES[action]} | reward={reward:+.1f}")
                self.pretty_print()
            else:
                print("  ⚠️  Commande invalide. Utilise 0/1/2/3.")

    # ------------------------------------------------------------------
    # Mode pas-à-pas
    # ------------------------------------------------------------------

    def play_policy_step_by_step(self, pi: np.ndarray, delay: float = 0.5):
        """
        Rejoue la policy apprise pas à pas, sans relancer l'apprentissage.

        Args:
            pi : matrice (25, 4) — policy apprise
            delay : pause entre chaque étape en mode auto (secondes)
        """
        ACTION_NAMES = {0: "Haut", 1: "Bas", 2: "Gauche", 3: "Droite"}

        print("\n🤖 MODE PAS-À-PAS — Replay de la policy")
        print("Appuie sur [Entrée] pour avancer, 'auto' pour automatique.\n")

        self.reset()
        self.pretty_print()

        step = 0
        mode_auto = False

        while not self.is_game_over():
            action = np.argmax(pi[self.current_state()])
            action_name = ACTION_NAMES[action]

            if not mode_auto:
                cmd = input(f"  Étape {step+1} — Action : {action_name} | [Entrée] / 'auto' : ").strip().lower()
                if cmd == "auto":
                    mode_auto = True
            else:
                print(f"  Étape {step+1} — Action : {action_name}")
                time.sleep(delay)

            prev_score = self.score()
            self.step(action)
            reward = self.score() - prev_score
            step += 1
            self.pretty_print()

            if step > 100:
                print("  ⚠️  Trop d'étapes — la policy boucle peut-être.")
                break

        print(f"\n  Terminé en {step} étape(s). Score final : {self.score():+.1f}")


# ----------------------------------------------------------------------
# Test rapide
# ----------------------------------------------------------------------
if __name__ == "__main__":
    env = GridWorldEnv()

    print("=== TEST EPISODE ALEATOIRE ===")
    env.reset()
    env.pretty_print()

    ACTION_NAMES = {0: "Haut", 1: "Bas", 2: "Gauche", 3: "Droite"}
    steps = 0

    while not env.is_game_over() and steps < 50:
        action = np.random.choice(env.available_actions())
        prev_score = env.score()
        env.step(action)
        reward = env.score() - prev_score
        print(f"  → {ACTION_NAMES[action]} | reward={reward:+.1f}")
        env.pretty_print()
        steps += 1
