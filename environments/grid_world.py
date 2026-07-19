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
        [A][ ][ ][ ][X]   <- état 4  = TRAP  (reward = -3.0, terminal)
        [ ][ ][ ][ ][ ]
        [ ][ ][ ][ ][ ]
        [ ][ ][ ][ ][ ]
        [ ][ ][ ][ ][G]   <- état 24 = GOAL  (reward = +1.0, terminal)

    Actions : 0=Haut, 1=Bas, 2=Gauche, 3=Droite
    Agent démarre en haut à gauche (état 0).
    """

    def __init__(self, rows: int = 5, cols: int = 5):
        self.rows = rows
        self.cols = cols
        self.trap = cols - 1
        self.goal = rows * cols - 1
        self.terminal_states = [self.trap, self.goal]
        self.agent_pos = 0
        self._score = 0.0

    def _state_to_rc(self, state: int):
        return state // self.cols, state % self.cols

    def _rc_to_state(self, row: int, col: int) -> int:
        return row * self.cols + col

    def _next_position(self, state: int, action: int) -> int:
        row, col = self._state_to_rc(state)
        if action == 0:
            row = max(row - 1, 0)
        elif action == 1:
            row = min(row + 1, self.rows - 1)
        elif action == 2:
            col = max(col - 1, 0)
        elif action == 3:
            col = min(col + 1, self.cols - 1)
        return self._rc_to_state(row, col)

    def _compute_reward(self, state: int) -> float:
        if state == self.goal:
            return 1.0
        elif state == self.trap:
            return -3.0
        return 0.0

    def reset(self):
        self.agent_pos = 0
        self._score = 0.0

    def step(self, action: int):
        if action not in self.available_actions():
            raise Exception(f"Action invalide : {action}")
        if self.is_game_over():
            raise Exception("Le jeu est terminé, appelle reset() d'abord !")
        self.agent_pos = self._next_position(self.agent_pos, action)
        self._score += self._compute_reward(self.agent_pos)

    def is_game_over(self) -> bool:
        return self.agent_pos in self.terminal_states

    def current_state(self) -> int:
        return self.agent_pos

    def available_actions(self) -> List[int]:
        return [0, 1, 2, 3]

    def score(self) -> float:
        return self._score

    def max_state_count(self) -> int:
        return self.rows * self.cols

    def max_actions_count(self) -> int:
        return 4

    def pretty_print(self):
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
                print("  Gagne ! Reward = +1.0")
            else:
                print("  Perdu ! Reward = -3.0")

    def play_human(self):
        print("\nMODE HUMAIN - Grid World")
        print("Commandes : [0]=Haut  [1]=Bas  [2]=Gauche  [3]=Droite  [q]=Quitter\n")
        self.reset()
        self.pretty_print()
        ACTION_NAMES = {0: "Haut", 1: "Bas", 2: "Gauche", 3: "Droite"}
        while not self.is_game_over():
            commande = input("  Ton action : ").strip().lower()
            if commande == "q":
                print("  Partie abandonnee.")
                break
            elif commande in ["0", "1", "2", "3"]:
                action = int(commande)
                prev_score = self.score()
                self.step(action)
                reward = self.score() - prev_score
                print(f"  -> {ACTION_NAMES[action]} | reward={reward:+.1f}")
                self.pretty_print()
            else:
                print("  Commande invalide. Utilise 0/1/2/3.")

    def play_policy_step_by_step(self, pi: np.ndarray, delay: float = 0.5):
        ACTION_NAMES = {0: "Haut", 1: "Bas", 2: "Gauche", 3: "Droite"}
        print("\nMODE PAS-A-PAS - Replay de la policy")
        print("Appuie sur [Entree] pour avancer, 'auto' pour automatique.\n")
        self.reset()
        self.pretty_print()
        step = 0
        mode_auto = False
        while not self.is_game_over():
            action = np.argmax(pi[self.current_state()])
            action_name = ACTION_NAMES[action]
            if not mode_auto:
                cmd = input(f"  Etape {step+1} - Action : {action_name} | [Entree] / 'auto' : ").strip().lower()
                if cmd == "auto":
                    mode_auto = True
            else:
                print(f"  Etape {step+1} - Action : {action_name}")
                time.sleep(delay)
            prev_score = self.score()
            self.step(action)
            reward = self.score() - prev_score
            step += 1
            self.pretty_print()
            if step > 100:
                print("  Trop d'etapes - la policy boucle peut-etre.")
                break
        print(f"\n  Termine en {step} etape(s). Score final : {self.score():+.1f}")


# ----------------------------------------------------------------------
# Construction du MDP (en dehors de la classe)
# ----------------------------------------------------------------------

def _build_gridworld_mdp():
    rows, cols = 5, 5
    trap, goal = 4, 24

    S = np.array(range(25))
    A = np.array([0, 1, 2, 3])
    R = np.array([0.0, -3.0, 1.0])
    T = np.array([trap, goal])

    def state_to_rc(s): return s // cols, s % cols
    def rc_to_state(r, c): return r * cols + c

    def next_state(s, a):
        r, c = state_to_rc(s)
        if a == 0: r = max(r-1, 0)
        elif a == 1: r = min(r+1, rows-1)
        elif a == 2: c = max(c-1, 0)
        elif a == 3: c = min(c+1, cols-1)
        return rc_to_state(r, c)

    def reward_index(s_next):
        if s_next == goal: return 2
        elif s_next == trap: return 1
        else: return 0

    p = np.zeros((25, 4, 25, 3))
    for s in range(25):
        if s in [trap, goal]:
            continue
        for a in range(4):
            s_next = next_state(s, a)
            r_idx = reward_index(s_next)
            p[s, a, s_next, r_idx] = 1.0

    return S, A, R, T, p


# Ajout du MDP a la classe
GridWorldEnv.S, GridWorldEnv.A, GridWorldEnv.R, GridWorldEnv.T, GridWorldEnv.p = _build_gridworld_mdp()


# ----------------------------------------------------------------------
# Test rapide
# ----------------------------------------------------------------------
if __name__ == "__main__":
    env = GridWorldEnv()

    print("=== TEST ===")
    env.reset()
    env.pretty_print()

    from algorithms.dynamic_programming import policy_iteration
    pi, V = policy_iteration(
        GridWorldEnv.S, GridWorldEnv.A, GridWorldEnv.R,
        GridWorldEnv.p, GridWorldEnv.T,
        gamma=0.99, theta=0.001
    )
    print("V:", V.reshape(5, 5).round(3))
