from typing import List
import numpy as np


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


class MontyHall2Env(ModelFreeEnv):
    """
    Environnement Monty Hall — Niveau 2 (5 portes, 4 actions).

    Règles :
        - 5 portes : 0, 1, 2, 3, 4
        - Une porte gagnante tirée aléatoirement (cachée à l'agent)
        - Étape 1 : agent choisit une porte parmi 5
        - Étapes 2, 3, 4 : le jeu retire une porte perdante
                           agent choisit 0=Garder ou 1=Changer
        - À la fin : 2 portes restantes, la choisie s'ouvre
        - Bonne porte → reward = +1.0
        - Mauvaise porte → reward = 0.0

    États :
        0          → étape 1 (choisir une porte parmi 5)
        1..5       → étape 2 (agent a choisi porte 0..4)
        6..10      → étape 3 (agent a gardé/changé)
        11..15     → étape 4 (agent a gardé/changé)
        16         → partie terminée
    """

    DOOR_NAMES = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    def __init__(self):
        self.winning_door = None       # porte gagnante cachée
        self.chosen_door = None        # porte actuelle de l'agent
        self.available_doors = None    # portes encore en jeu
        self.removed_door = None       # dernière porte retirée
        self.step_number = None        # étape actuelle (1, 2, 3, 4)
        self._score = 0.0
        self.game_over = False

    # ------------------------------------------------------------------
    # Helper interne
    # ------------------------------------------------------------------

    def _remove_one_losing_door(self):
        """Le jeu retire une porte perdante parmi les disponibles (non choisie)."""
        losing_doors = [d for d in self.available_doors
                        if d != self.chosen_door
                        and d != self.winning_door]
        self.removed_door = np.random.choice(losing_doors)
        self.available_doors.remove(self.removed_door)

    # ------------------------------------------------------------------
    # Contrat ModelFreeEnv
    # ------------------------------------------------------------------

    def reset(self):
        """Réinitialise l'environnement."""
        self.winning_door = np.random.randint(0, 5)
        self.chosen_door = None
        self.available_doors = [0, 1, 2, 3, 4]
        self.removed_door = None
        self.step_number = 1
        self._score = 0.0
        self.game_over = False

    def step(self, action: int):
        """
        Joue une action. Ne retourne rien.

        Étape 1 : action = porte choisie (0..4)
        Étapes 2,3,4 : action = 0=Garder, 1=Changer
        """
        if action not in self.available_actions():
            raise Exception(f"Action invalide : {action}")
        if self.is_game_over():
            raise Exception("La partie est terminée, appelle reset() d'abord !")

        if self.step_number == 1:
            # Agent choisit une porte parmi 5
            self.chosen_door = action
            # Le jeu retire une porte perdante
            self._remove_one_losing_door()
            self.step_number = 2

        elif self.step_number in [2, 3]:
            # 0 = Garder, 1 = Changer
            if action == 1:
                # Changer → prendre une autre porte disponible aléatoirement
                other_doors = [d for d in self.available_doors if d != self.chosen_door]
                self.chosen_door = np.random.choice(other_doors)

            # Le jeu retire encore une porte perdante
            self._remove_one_losing_door()
            self.step_number += 1

        elif self.step_number == 4:
            # Dernière décision : Garder ou Changer
            if action == 1:
                other_doors = [d for d in self.available_doors if d != self.chosen_door]
                self.chosen_door = other_doors[0]

            # Résultat final
            self._score = 1.0 if self.chosen_door == self.winning_door else 0.0
            self.game_over = True

    def is_game_over(self) -> bool:
        """Retourne True si la partie est terminée."""
        return self.game_over

    def current_state(self) -> int:
        """
        Retourne l'état actuel :
            0      → étape 1
            1..5   → étape 2, agent a choisi porte 0..4
            6..10  → étape 3
            11..15 → étape 4
            16     → terminé
        """
        if self.game_over:
            return 16
        if self.step_number == 1:
            return 0
        offset = (self.step_number - 2) * 5 + 1
        return offset + self.chosen_door

    def available_actions(self) -> List[int]:
        """
        Étape 1 : [0, 1, 2, 3, 4] → choisir une porte
        Étapes 2,3,4 : [0, 1] → Garder ou Changer
        """
        if self.step_number == 1:
            return [0, 1, 2, 3, 4]
        return [0, 1]

    def score(self) -> float:
        """Retourne le score."""
        return self._score

    def max_state_count(self) -> int:
        """17 états possibles."""
        return 17

    def max_actions_count(self) -> int:
        """5 actions max (étape 1)."""
        return 5

    def pretty_print(self):
        """Affiche l'état actuel."""
        print()
        print("  🚪 Monty Hall — Niveau 2 (5 portes)")
        print(f"  {'─' * 40}")

        # Affiche les portes disponibles
        doors_display = ""
        for d in range(5):
            if d not in self.available_doors:
                doors_display += f" [╳]"   # porte retirée
            elif d == self.chosen_door:
                doors_display += f" [{self.DOOR_NAMES[d]}*]"  # porte choisie
            else:
                doors_display += f" [{self.DOOR_NAMES[d]}]"   # porte disponible

        print(f"  Portes : {doors_display}")
        print(f"  Porte choisie : {self.DOOR_NAMES[self.chosen_door] if self.chosen_door is not None else '???'}")

        if self.step_number == 1:
            print("  Étape 1 — Choisissez une porte [0=A, 1=B, 2=C, 3=D, 4=E]")

        elif self.step_number in [2, 3, 4] and not self.game_over:
            print(f"  Le jeu a retiré la porte {self.DOOR_NAMES[self.removed_door]} (perdante)")
            other = [self.DOOR_NAMES[d] for d in self.available_doors if d != self.chosen_door]
            print(f"  Étape {self.step_number} — [0]=Garder {self.DOOR_NAMES[self.chosen_door]}  [1]=Changer pour {other}")

        elif self.game_over:
            print(f"  Porte gagnante était : {self.DOOR_NAMES[self.winning_door]}")
            if self._score == 1.0:
                print("  ✅ Bonne porte ! reward = +1.0")
            else:
                print("  ❌ Mauvaise porte ! reward = 0.0")

        print(f"  {'─' * 40}")

    # ------------------------------------------------------------------
    # Mode humain
    # ------------------------------------------------------------------

    def play_human(self):
        """Mode interactif : l'utilisateur joue manuellement."""
        print("\n🎮 MODE HUMAIN — Monty Hall Niveau 2")

        self.reset()
        self.pretty_print()

        while not self.is_game_over():
            if self.step_number == 1:
                prompt = "  Étape 1 — Choisis une porte [0=A, 1=B, 2=C, 3=D, 4=E] : "
            else:
                prompt = f"  Étape {self.step_number} — [0]=Garder  [1]=Changer : "

            commande = input(prompt).strip()

            if commande == "q":
                print("  Partie abandonnée.")
                break

            valid = [str(a) for a in self.available_actions()]
            if commande in valid:
                self.step(int(commande))
                self.pretty_print()
            else:
                print(f"  ⚠️  Commande invalide. Choix possibles : {valid}")

    # ------------------------------------------------------------------
    # Mode pas-à-pas
    # ------------------------------------------------------------------

    def play_policy_step_by_step(self, pi: np.ndarray):
        """
        Rejoue la policy apprise pas à pas.

        Args:
            pi : matrice (17, 5) — policy apprise
        """
        print("\n🤖 MODE PAS-À-PAS — Replay de la policy")

        self.reset()
        self.pretty_print()

        while not self.is_game_over():
            state = self.current_state()
            action = np.argmax(pi[state])

            if self.step_number == 1:
                action_name = f"Porte {self.DOOR_NAMES[action]}"
            else:
                action_name = "Garder" if action == 0 else "Changer"

            input(f"  Étape {self.step_number} — Action : {action_name} | [Entrée] pour jouer : ")
            self.step(action)
            self.pretty_print()

        print(f"\n  Score final : {self.score():+.1f}")


# ----------------------------------------------------------------------
# Test rapide
# ----------------------------------------------------------------------
if __name__ == "__main__":

    env = MontyHall2Env()

    print("=== TEST EPISODE ALEATOIRE ===")
    env.reset()
    env.pretty_print()

    while not env.is_game_over():
        action = np.random.choice(env.available_actions())
        env.step(action)
        env.pretty_print()

    print("\n=== TEST STRATEGIE OPTIMALE — Toujours Changer (1000 parties) ===")
    wins = 0
    for _ in range(1000):
        env.reset()
        env.step(np.random.randint(0, 5))  # étape 1 : porte aléatoire
        env.step(1)                         # étape 2 : changer
        env.step(1)                         # étape 3 : changer
        env.step(1)                         # étape 4 : changer
        if env.score() == 1.0:
            wins += 1
    print(f"  Victoires en changeant toujours : {wins}/1000")

    print("\n=== TEST STRATEGIE — Toujours Garder (1000 parties) ===")
    wins = 0
    for _ in range(1000):
        env.reset()
        env.step(np.random.randint(0, 5))  # étape 1 : porte aléatoire
        env.step(0)                         # étape 2 : garder
        env.step(0)                         # étape 3 : garder
        env.step(0)                         # étape 4 : garder
        if env.score() == 1.0:
            wins += 1
    print(f"  Victoires en gardant toujours : {wins}/1000")
