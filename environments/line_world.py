from typing import List
import numpy as np


class ModelFreeEnv:
    """
    Contrat de base pour tous les environnements Model-Free.
    Tous les environnements DOIVENT implémenter ces méthodes.
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


class LineWorldEnv(ModelFreeEnv):
    """
    Environnement Line World.

    Représentation :
        [X] [ ] [ ] [ ] [★]
         0    1    2    3    4

    - État 0 : terminal perdant (score = -1.0)
    - État 4 : terminal gagnant (score = +1.0)
    - Actions : 0 = Gauche, 1 = Droite
    """

    def __init__(self, num_cells: int = 5):
        self.num_cells = num_cells
        self.agent_pos = self.num_cells // 2  # départ au milieu

    # ------------------------------------------------------------------
    # Contrat ModelFreeEnv
    # ------------------------------------------------------------------

    def reset(self):
        """Réinitialise l'environnement."""
        self.agent_pos = self.num_cells // 2

    def step(self, action: int):
        """
        Joue une action. Ne retourne rien.
        Utilise current_state(), score(), is_game_over() pour récupérer les infos.
        """
        if action not in self.available_actions():
            raise Exception("Action invalide !")
        if self.is_game_over():
            raise Exception("Le jeu est terminé, appelle reset() d'abord !")

        if action == 0:
            self.agent_pos -= 1
        elif action == 1:
            self.agent_pos += 1

    def is_game_over(self) -> bool:
        """Retourne True si l'agent est sur un état terminal."""
        return self.agent_pos == 0 or self.agent_pos == self.num_cells - 1

    def current_state(self) -> int:
        """Retourne la position actuelle de l'agent."""
        return self.agent_pos

    def available_actions(self) -> List[int]:
        """Retourne les actions disponibles : 0=Gauche, 1=Droite."""
        return [0, 1]

    def score(self) -> float:
        """Retourne le score actuel de l'environnement."""
        if self.agent_pos == 0:
            return -1.0
        elif self.agent_pos == self.num_cells - 1:
            return 1.0
        return 0.0

    def max_state_count(self) -> int:
        """Nombre total d'états possibles."""
        return self.num_cells

    def max_actions_count(self) -> int:
        """Nombre total d'actions possibles."""
        return 2

    def pretty_print(self):
        """Affiche l'état actuel de l'environnement."""
        line = ""
        for i in range(self.num_cells):
            if i == self.agent_pos:
                line += "[A]"
            elif i == 0:
                line += "[X]"
            elif i == self.num_cells - 1:
                line += "[★]"
            else:
                line += "[ ]"
        print(line)
        if self.is_game_over():
            if self.agent_pos == self.num_cells - 1:
                print("✅ Gagné ! Score = +1.0")
            else:
                print("❌ Perdu ! Score = -1.0")

    # ------------------------------------------------------------------
    # Mode humain
    # ------------------------------------------------------------------

    def play_human(self):
        """Mode interactif : l'utilisateur joue manuellement."""
        print("\n🎮 MODE HUMAIN — Line World")
        print("Commandes : [0] Gauche  |  [1] Droite  |  [q] Quitter\n")

        self.reset()
        self.pretty_print()

        while not self.is_game_over():
            commande = input("Ton action : ").strip().lower()

            if commande == "q":
                print("Partie abandonnée.")
                break
            elif commande in ["0", "1"]:
                prev_score = self.score()
                self.step(int(commande))
                reward = self.score() - prev_score
                action_name = "Gauche" if commande == "0" else "Droite"
                print(f"→ {action_name} | reward={reward:+.1f}")
                self.pretty_print()
            else:
                print("⚠️  Commande invalide. Tape 0 (Gauche) ou 1 (Droite).")

    # ------------------------------------------------------------------
    # Mode pas-à-pas
    # ------------------------------------------------------------------

    def play_policy_step_by_step(self, pi: np.ndarray, delay: float = 0.0):
        """
        Rejoue la policy apprise pas à pas, sans relancer l'apprentissage.

        Args:
            pi: matrice (num_cells, 2) — policy apprise
            delay: pause entre chaque étape (secondes)
        """
        import time
        print("\n🤖 MODE PAS-À-PAS — Replay de la policy")
        print("Appuie sur [Entrée] pour avancer, 'auto' pour automatique.\n")

        self.reset()
        self.pretty_print()

        step = 0
        mode_auto = False

        while not self.is_game_over():
            action = np.argmax(pi[self.current_state()])
            action_name = "Gauche" if action == 0 else "Droite"

            if not mode_auto:
                cmd = input(f"Étape {step+1} — Action : {action_name} | [Entrée] / 'auto' : ").strip().lower()
                if cmd == "auto":
                    mode_auto = True
            else:
                print(f"Étape {step+1} — Action : {action_name}")
                time.sleep(delay)

            prev_score = self.score()
            self.step(action)
            reward = self.score() - prev_score
            step += 1
            self.pretty_print()

        print(f"\nTerminé en {step} étape(s). Score final : {self.score()}")


# ----------------------------------------------------------------------
# Test rapide
# ----------------------------------------------------------------------
if __name__ == "__main__":
    env = LineWorldEnv()

    print("=== TEST EPISODE ALEATOIRE ===")
    env.reset()
    env.pretty_print()

    while not env.is_game_over():
        action = np.random.choice(env.available_actions())
        prev_score = env.score()
        env.step(action)
        reward = env.score() - prev_score
        print(f"→ {'Gauche' if action == 0 else 'Droite'} | reward={reward:+.1f}")
        env.pretty_print()

    # Mode humain — décommente pour jouer :
    # env.play_human()
