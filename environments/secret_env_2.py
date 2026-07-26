from typing import List
import numpy as np

from environments.secret_envs_wrapper import SecretEnv2Raw


class ModelFreeEnv:
    """
    Contrat de base pour tous les environnements Model-Free.
    """
    def reset(self): raise NotImplementedError
    def step(self, action: int): raise NotImplementedError
    def is_game_over(self) -> bool: raise NotImplementedError
    def current_state(self) -> int: raise NotImplementedError
    def available_actions(self) -> List[int]: raise NotImplementedError
    def score(self) -> float: raise NotImplementedError
    def max_state_count(self) -> int: raise NotImplementedError
    def max_actions_count(self) -> int: raise NotImplementedError
    def pretty_print(self): raise NotImplementedError


class SecretEnv2(ModelFreeEnv):
    """
    Adapter pour Secret Env 2 fourni par le prof.

    L'environnement est boite noire : on ne connait ni sa structure ni sa
    semantique. On expose l'interface ModelFreeEnv (comme LineWorld / GridWorld)
    ainsi que le modele MDP (S, A, R, T, p) pour pouvoir utiliser DP.
    """

    def __init__(self):
        self._raw = SecretEnv2Raw()
        self._nb_states = self._raw.num_states()
        self._nb_actions = self._raw.num_actions()
        self._nb_rewards = self._raw.num_rewards()

    # ------------------------------------------------------------------
    # Contrat ModelFreeEnv
    # ------------------------------------------------------------------

    def reset(self):
        self._raw.reset()

    def step(self, action: int):
        if self._raw.is_game_over():
            raise Exception("Le jeu est termine, appelle reset() d'abord.")
        self._raw.step(int(action))

    def is_game_over(self) -> bool:
        return bool(self._raw.is_game_over())

    def current_state(self) -> int:
        return int(self._raw.state_id())

    def available_actions(self) -> List[int]:
        return [int(a) for a in self._raw.available_actions()]

    def score(self) -> float:
        return float(self._raw.score())

    def max_state_count(self) -> int:
        return self._nb_states

    def max_actions_count(self) -> int:
        return self._nb_actions

    def pretty_print(self):
        print(f"[SecretEnv2] state={self.current_state()} "
              f"score={self.score():+.3f} "
              f"available={self.available_actions()} "
              f"terminal={self.is_game_over()}")
        try:
            self._raw.display()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Mode humain
    # ------------------------------------------------------------------

    def play_human(self):
        print("\nMODE HUMAIN - Secret Env 2")
        print("Tape le numero d'action, 'q' pour quitter.\n")
        self.reset()
        self.pretty_print()
        while not self.is_game_over():
            available = self.available_actions()
            cmd = input(f"  Actions dispo {available} : ").strip().lower()
            if cmd == "q":
                print("  Abandon.")
                break
            try:
                a = int(cmd)
            except ValueError:
                print("  Commande invalide.")
                continue
            if a not in available:
                print("  Action non disponible.")
                continue
            prev = self.score()
            self.step(a)
            print(f"  -> action {a} | reward={self.score() - prev:+.3f}")
            self.pretty_print()
        print(f"  Score final : {self.score():+.3f}")

    # ------------------------------------------------------------------
    # Mode pas-a-pas (rejoue une policy apprise)
    # ------------------------------------------------------------------

    def play_policy_step_by_step(self, pi: np.ndarray, delay: float = 0.0):
        import time
        print("\nMODE PAS-A-PAS - Replay de la policy sur Secret Env 2\n")
        self.reset()
        self.pretty_print()
        step = 0
        mode_auto = False
        while not self.is_game_over() and step < 500:
            s = self.current_state()
            available = self.available_actions()
            # argmax restreint aux actions disponibles
            action = max(available, key=lambda a: pi[s, a])
            if not mode_auto:
                cmd = input(f"  Etape {step+1} - Action : {action} | [Entree] / 'auto' : ").strip().lower()
                if cmd == "auto":
                    mode_auto = True
            else:
                print(f"  Etape {step+1} - Action : {action}")
                time.sleep(delay)
            prev = self.score()
            self.step(action)
            print(f"    reward = {self.score() - prev:+.3f}")
            self.pretty_print()
            step += 1
        print(f"\n  Termine en {step} etape(s). Score final : {self.score():+.3f}")


# ----------------------------------------------------------------------
# Construction du modele MDP (S, A, R, T, p)
# ----------------------------------------------------------------------

def build_secret_env_2_mdp(raw: SecretEnv2Raw = None, verbose: bool = True):
    """
    Construit le modele MDP complet pour Secret Env 2 :
        - S : indices des etats [0..num_states-1]
        - A : indices des actions [0..num_actions-1]
        - R : valeurs des rewards possibles
        - p : matrice de transition p[s, a, s', r_index]
        - T : etats terminaux (heuristique)

    ATTENTION : la matrice p a une taille |S|*|A|*|S|*|R|. Pour un env avec
    par exemple 1000 etats, 4 actions, 3 rewards, ca fait 12M floats = ~96 Mo.
    Verifie les dimensions imprimees avant d'appeler cette fonction sur un env
    a gros espace d'etats.
    """
    if raw is None:
        raw = SecretEnv2Raw()

    nb_s = raw.num_states()
    nb_a = raw.num_actions()
    nb_r = raw.num_rewards()

    if verbose:
        print(f"[SecretEnv2] nb_states={nb_s} nb_actions={nb_a} nb_rewards={nb_r}")
        print(f"[SecretEnv2] Taille matrice p : {nb_s*nb_a*nb_s*nb_r:,} floats "
              f"(~{nb_s*nb_a*nb_s*nb_r*8/1e6:.1f} Mo en float64)")

    S = np.arange(nb_s)
    A = np.arange(nb_a)
    R = np.array([raw.reward(i) for i in range(nb_r)], dtype=np.float64)

    p = np.zeros((nb_s, nb_a, nb_s, nb_r), dtype=np.float64)
    for s in range(nb_s):
        for a in range(nb_a):
            for s_p in range(nb_s):
                for r in range(nb_r):
                    p[s, a, s_p, r] = raw.p(s, a, s_p, r)

    # Heuristique : un etat est terminal si toutes ses transitions sortantes
    # ont une probabilite nulle (pas de sortie possible).
    T_list = []
    for s in range(nb_s):
        if p[s].sum() == 0.0:
            T_list.append(s)
    T = np.array(T_list, dtype=np.int64)

    if verbose:
        print(f"[SecretEnv2] rewards possibles = {R.tolist()}")
        print(f"[SecretEnv2] {len(T)} etats terminaux detectes (heuristique)")

    return S, A, R, T, p


# ----------------------------------------------------------------------
# Test rapide
# ----------------------------------------------------------------------
if __name__ == "__main__":
    env = SecretEnv2()
    print(f"num_states={env.max_state_count()} num_actions={env.max_actions_count()}")
    env.reset()
    env.pretty_print()
    print("Actions dispo :", env.available_actions())

    step = 0
    while not env.is_game_over() and step < 20:
        a = np.random.choice(env.available_actions())
        prev = env.score()
        env.step(a)
        step += 1
        print(f"step {step} action={a} reward={env.score()-prev:+.3f}")
    print("Score final :", env.score())
