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


class MontyHall1Env(ModelFreeEnv):
    """
    Environnement Monty Hall — Niveau 1 (3 portes, 2 actions).

    Règles :
        - 3 portes : 0=A, 1=B, 2=C
        - Une porte gagnante tirée aléatoirement (cachée à l'agent)
        - Étape 1 : l'agent choisit une porte parmi 0, 1, 2
        - Le jeu retire une porte PERDANTE parmi les non-choisies
        - Étape 2 : l'agent choisit 0=Garder ou 1=Changer
        - Bonne porte → reward = +1.0
        - Mauvaise porte → reward = 0.0

    États :
        0        → début, étape 1 (choisir une porte)
        1, 2, 3  → étape 2, agent a choisi la porte 0/1/2 à l'étape 1
        4        → partie terminée
    """

    DOOR_NAMES = {0: "A", 1: "B", 2: "C"}

    def __init__(self):
        self.winning_door = None      # porte gagnante (cachée)
        self.chosen_door = None       # porte choisie par l'agent à l'étape 1
        self.removed_door = None      # porte retirée par le jeu
        self.remaining_door = None    # porte restante après retrait
        self.step_number = None       # étape actuelle (1 ou 2)
        self._score = 0.0
        self.game_over = False

    # ------------------------------------------------------------------
    # Contrat ModelFreeEnv
    # ------------------------------------------------------------------

    def reset(self):
        """Réinitialise l'environnement."""
        # Tire aléatoirement la porte gagnante (cachée à l'agent)
        self.winning_door = np.random.randint(0, 3)
        self.chosen_door = None
        self.removed_door = None
        self.remaining_door = None
        self.step_number = 1
        self._score = 0.0
        self.game_over = False

    def step(self, action: int):
        """
        Joue une action. Ne retourne rien.

        Étape 1 : action = porte choisie (0, 1 ou 2)
        Étape 2 : action = 0=Garder, 1=Changer
        """
        if action not in self.available_actions():
            raise Exception(f"Action invalide : {action}")
        if self.is_game_over():
            raise Exception("La partie est terminée, appelle reset() d'abord !")

        if self.step_number == 1:
            # L'agent choisit une porte
            self.chosen_door = action

            # Le jeu retire une porte perdante parmi les non-choisies
            other_doors = [d for d in [0, 1, 2] if d != self.chosen_door]
            losing_doors = [d for d in other_doors if d != self.winning_door]
            self.removed_door = np.random.choice(losing_doors)

            # La porte restante (ni choisie, ni retirée)
            self.remaining_door = [d for d in [0, 1, 2]
                                   if d != self.chosen_door
                                   and d != self.removed_door][0]
            self.step_number = 2

        elif self.step_number == 2:
            # action 0 = Garder, action 1 = Changer
            if action == 0:
                final_door = self.chosen_door
            else:
                final_door = self.remaining_door

            # Résultat
            if final_door == self.winning_door:
                self._score = 1.0
            else:
                self._score = 0.0

            self.game_over = True

    def is_game_over(self) -> bool:
        """Retourne True si la partie est terminée."""
        return self.game_over

    def current_state(self) -> int:
        """
        Retourne l'état actuel :
            0        → étape 1, rien choisi encore
            1, 2, 3  → étape 2, agent a choisi porte 0/1/2
            4        → partie terminée
        """
        if self.game_over:
            return 4
        if self.step_number == 1:
            return 0
        return self.chosen_door + 1

    def available_actions(self) -> List[int]:
        """
        Étape 1 : [0, 1, 2] → choisir une porte
        Étape 2 : [0, 1]    → 0=Garder, 1=Changer
        """
        if self.step_number == 1:
            return [0, 1, 2]
        return [0, 1]

    def score(self) -> float:
        """Retourne le score."""
        return self._score

    def max_state_count(self) -> int:
        """5 états : 0, 1, 2, 3, 4."""
        return 5

    def max_actions_count(self) -> int:
        """3 actions max (étape 1)."""
        return 3

    def pretty_print(self):
        """Affiche l'état actuel."""
        print()
        print("  🚪 Monty Hall — Niveau 1 (3 portes)")
        print(f"  {'─' * 35}")

        if self.step_number == 1:
            print("  Étape 1 — Choisissez une porte !")
            print("  Portes : [0]=A  [1]=B  [2]=C")
            print("  Porte gagnante : ???")

        elif self.step_number == 2 and not self.game_over:
            print(f"  Étape 1 : Tu as choisi la porte {self.DOOR_NAMES[self.chosen_door]}")
            print(f"  Le jeu retire la porte {self.DOOR_NAMES[self.removed_door]} (perdante)")
            print(f"  Étape 2 — Garder {self.DOOR_NAMES[self.chosen_door]} ou Changer pour {self.DOOR_NAMES[self.remaining_door]} ?")
            print("  Actions : [0]=Garder  [1]=Changer")

        elif self.game_over:
            print(f"  Porte gagnante était : {self.DOOR_NAMES[self.winning_door]}")
            if self._score == 1.0:
                print("  ✅ Bonne porte ! reward = +1.0")
            else:
                print("  ❌ Mauvaise porte ! reward = 0.0")

        print(f"  {'─' * 35}")

    # ------------------------------------------------------------------
    # Mode humain
    # ------------------------------------------------------------------

    def play_human(self):
        """Mode interactif : l'utilisateur joue manuellement."""
        print("\n🎮 MODE HUMAIN — Monty Hall Niveau 1")

        self.reset()
        self.pretty_print()

        while not self.is_game_over():
            if self.step_number == 1:
                prompt = "  Étape 1 — Choisis une porte [0=A, 1=B, 2=C] : "
            else:
                prompt = "  Étape 2 — [0]=Garder  [1]=Changer : "

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
            pi : matrice (5, 3) — policy apprise
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
        
def _build_monty_hall_1_mdp():
    """
    Construit le MDP complet de Monty Hall niveau 1.

    États : 0=départ, 1/2/3=porte A/B/C choisie (étape 2), 4=terminé
    Actions : à l'état 0, choisir une porte (0,1,2) ; aux états 1,2,3, 0=garder / 1=changer
    """
    S = np.array([0, 1, 2, 3, 4])
    A = np.array([0, 1, 2])   # max 3 actions (utilisées seulement à l'état 0)
    R = np.array([0.0, 1.0])  # R[0]=perdu, R[1]=gagné
    T = np.array([4])

    p = np.zeros((len(S), len(A), len(S), len(R)))

    # État 0 : choisir une porte -> état 1/2/3, reward 0 (rien décidé encore)
    for a in range(3):
        p[0, a, a + 1, 0] = 1.0

    # États 1, 2, 3 : garder (action 0) ou changer (action 1)
    for s in [1, 2, 3]:
        p[s, 0, 4, 1] = 1 / 3   # garder -> gagne 1 fois sur 3
        p[s, 0, 4, 0] = 2 / 3   # garder -> perd 2 fois sur 3
        p[s, 1, 4, 1] = 2 / 3   # changer -> gagne 2 fois sur 3
        p[s, 1, 4, 0] = 1 / 3   # changer -> perd 1 fois sur 3

    return S, A, R, T, p


MontyHall1Env.S, MontyHall1Env.A, MontyHall1Env.R, MontyHall1Env.T, MontyHall1Env.p = _build_monty_hall_1_mdp()


# ----------------------------------------------------------------------
# Test rapide
# ----------------------------------------------------------------------
if __name__ == "__main__":

    env = MontyHall1Env()

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
        env.step(np.random.randint(0, 3))  # étape 1 : porte aléatoire
        env.step(1)                         # étape 2 : toujours changer
        if env.score() == 1.0:
            wins += 1
    print(f"  Victoires en changeant : {wins}/1000 (~666 attendu, soit ~66.7%)")

    print("\n=== TEST STRATEGIE SOUS-OPTIMALE — Toujours Garder (1000 parties) ===")
    wins = 0
    for _ in range(1000):
        env.reset()
        env.step(np.random.randint(0, 3))  # étape 1 : porte aléatoire
        env.step(0)                         # étape 2 : toujours garder
        if env.score() == 1.0:
            wins += 1
    print(f"  Victoires en gardant : {wins}/1000 (~333 attendu, soit ~33.3%)")
