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


class RockPaperScissorsEnv(ModelFreeEnv):
    """
    Environnement Pierre Feuille Ciseaux — 2 rounds.

    Règles :
        - Round 1 : l'adversaire joue ALÉATOIREMENT
        - Round 2 : l'adversaire joue FORCÉMENT ce que l'agent a joué au round 1
        - Gagner un round  → reward = +1
        - Perdre un round  → reward = -1
        - Égalité          → reward =  0

    Actions : 0=Pierre, 1=Feuille, 2=Ciseaux

    États :
        0        → début de partie, round 1
        1, 2, 3  → round 2, l'agent a joué Pierre/Feuille/Ciseaux au round 1
        4        → partie terminée
    """

    ACTION_NAMES = {0: "Pierre", 1: "Feuille", 2: "Ciseaux"}

    # Ce qui bat quoi : BEATS[a] = action battue par a
    # Pierre(0) bat Ciseaux(2)
    # Feuille(1) bat Pierre(0)
    # Ciseaux(2) bat Feuille(1)
    BEATS = {0: 2, 1: 0, 2: 1}

    def __init__(self):
        self.round = None           # round actuel (1 ou 2)
        self.agent_action_r1 = None # action de l'agent au round 1
        self.adv_action_r1 = None   # action de l'adversaire au round 1
        self._score = 0.0           # score cumulé
        self.game_over = False

    # ------------------------------------------------------------------
    # Helpers internes
    # ------------------------------------------------------------------

    def _round_result(self, agent_action: int, adv_action: int) -> float:
        """Calcule le reward d'un round."""
        if agent_action == adv_action:
            return 0.0                          # égalité
        elif self.BEATS[agent_action] == adv_action:
            return 1.0                          # agent gagne
        else:
            return -1.0                         # agent perd

    # ------------------------------------------------------------------
    # Contrat ModelFreeEnv
    # ------------------------------------------------------------------

    def reset(self):
        """Réinitialise l'environnement."""
        self.round = 1
        self.agent_action_r1 = None
        self.adv_action_r1 = None
        self._score = 0.0
        self.game_over = False

    def step(self, action: int):
        """
        Joue une action. Ne retourne rien.
        Utilise current_state(), score(), is_game_over() pour les infos.
        """
        if action not in self.available_actions():
            raise Exception(f"Action invalide : {action}")
        if self.is_game_over():
            raise Exception("La partie est terminée, appelle reset() d'abord !")

        if self.round == 1:
            # Adversaire joue aléatoirement
            self.adv_action_r1 = np.random.randint(0, 3)
            self.agent_action_r1 = action

            reward = self._round_result(action, self.adv_action_r1)
            self._score += reward
            self.round = 2

        elif self.round == 2:
            # Adversaire joue FORCÉMENT ce que l'agent a joué au round 1
            adv_action_r2 = self.agent_action_r1

            reward = self._round_result(action, adv_action_r2)
            self._score += reward
            self.game_over = True

    def is_game_over(self) -> bool:
        """Retourne True si les 2 rounds sont joués."""
        return self.game_over

    def current_state(self) -> int:
        """
        Retourne l'état actuel :
            0        → début, round 1
            1, 2, 3  → round 2, agent a joué Pierre/Feuille/Ciseaux au round 1
            4        → partie terminée
        """
        if self.game_over:
            return 4
        if self.round == 1:
            return 0
        # round 2 : état = action du round 1 + 1
        return self.agent_action_r1 + 1

    def available_actions(self) -> List[int]:
        """Retourne les actions disponibles : 0=Pierre, 1=Feuille, 2=Ciseaux."""
        return [0, 1, 2]

    def score(self) -> float:
        """Retourne le score cumulé."""
        return self._score

    def max_state_count(self) -> int:
        """5 états possibles : 0, 1, 2, 3, 4."""
        return 5

    def max_actions_count(self) -> int:
        """3 actions : Pierre, Feuille, Ciseaux."""
        return 3

    def pretty_print(self):
        """Affiche l'état actuel de la partie."""
        print()
        print("  ✊ Pierre Feuille Ciseaux — 2 rounds")
        print(f"  {'─' * 35}")

        if self.round == 1 and not self.game_over:
            print("  Round 1 — Choisissez votre action !")
            print("  Adversaire : ???")

        elif self.round == 2 and not self.game_over:
            # Affiche le résultat du round 1
            r1_result = self._round_result(self.agent_action_r1, self.adv_action_r1)
            r1_emoji = "✅" if r1_result > 0 else ("❌" if r1_result < 0 else "🤝")
            print(f"  Round 1 : Toi={self.ACTION_NAMES[self.agent_action_r1]}"
                  f" | Adv={self.ACTION_NAMES[self.adv_action_r1]}"
                  f" | {r1_emoji} reward={r1_result:+.0f}")
            print(f"  Round 2 — Adversaire jouera : {self.ACTION_NAMES[self.agent_action_r1]} (copie ton R1)")

        elif self.game_over:
            print(f"  Round 1 : Toi={self.ACTION_NAMES[self.agent_action_r1]}"
                  f" | Adv={self.ACTION_NAMES[self.adv_action_r1]}")
            print(f"  Round 2 : terminé")
            print(f"  {'─' * 35}")
            if self._score > 0:
                print(f"  🏆 Score final : {self._score:+.0f} — Victoire !")
            elif self._score < 0:
                print(f"  💀 Score final : {self._score:+.0f} — Défaite !")
            else:
                print(f"  🤝 Score final : {self._score:+.0f} — Égalité !")

        print(f"  {'─' * 35}")

    # ------------------------------------------------------------------
    # Mode humain
    # ------------------------------------------------------------------

    def play_human(self):
        """Mode interactif : l'utilisateur joue manuellement."""
        print("\n🎮 MODE HUMAIN — Pierre Feuille Ciseaux 2 rounds")
        print("Actions : [0]=Pierre  [1]=Feuille  [2]=Ciseaux  [q]=Quitter\n")

        self.reset()
        self.pretty_print()

        while not self.is_game_over():
            commande = input(f"  Round {self.round} — Ton action : ").strip().lower()

            if commande == "q":
                print("  Partie abandonnée.")
                break
            elif commande in ["0", "1", "2"]:
                self.step(int(commande))
                self.pretty_print()
            else:
                print("  ⚠️  Commande invalide. Tape 0, 1 ou 2.")

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
            input(f"  Round {self.round} — Action choisie : {self.ACTION_NAMES[action]} | [Entrée] pour jouer : ")
            self.step(action)
            self.pretty_print()

        print(f"\n  Score final : {self.score():+.0f}")


# ----------------------------------------------------------------------
# Test rapide
# ----------------------------------------------------------------------
if __name__ == "__main__":

    print("=== TEST EPISODE ALEATOIRE ===")
    env = RockPaperScissorsEnv()
    env.reset()
    env.pretty_print()

    while not env.is_game_over():
        action = np.random.choice(env.available_actions())
        env.step(action)
        env.pretty_print()

    print("\n=== TEST STRATEGIE OPTIMALE (5 parties) ===")
    # Stratégie optimale : au R2 l'adversaire copie ton R1
    # → joue ce qui bat ton R1 au R2
    BEATS = {0: 1, 1: 2, 2: 0}  # Pierre→Feuille, Feuille→Ciseaux, Ciseaux→Pierre
    r2_wins = 0
    for i in range(5):
        env.reset()
        r1_action = np.random.randint(0, 3)   # R1 : joue n'importe quoi
        env.step(r1_action)
        r2_action = BEATS[r1_action]           # R2 : joue ce qui bat R1
        score_before_r2 = env.score()
        env.step(r2_action)
        r2_reward = env.score() - score_before_r2
        print(f"  Partie {i+1} | R1={env.ACTION_NAMES[r1_action]} | R2={env.ACTION_NAMES[r2_action]} | R2 reward={r2_reward:+.0f} | Score total={env.score():+.0f}")
        if r2_reward > 0:
            r2_wins += 1
    print(f"\n  Victoires Round 2 : {r2_wins}/5 (doit être 5/5 !)")
