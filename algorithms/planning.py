import numpy as np
from typing import List
from environments.line_world import LineWorldEnv  # pour le type hint


def choose_action_epsilon_greedy(
    Q: np.ndarray,
    s: int,
    epsilon: float,
    available_actions: List[int]
) -> int:
    if np.random.random() < epsilon:
        return np.random.choice(available_actions)
    # Prend le max uniquement parmi les actions disponibles
    return max(available_actions, key=lambda a: Q[s, a])

def dyna_q(
    env,
    gamma: float = 0.999,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    max_steps: int = 10_000,
    N: int = 10,
    cumulated_rewards: List[float] = None
):
    """
    Algorithme Dyna-Q.

    Combine Q-Learning (apprentissage réel) et planification
    (simulation d'expériences passées).

    Args:
        env     : environnement (ModelFreeEnv)
        gamma   : facteur de discount
        alpha   : taux d'apprentissage
        epsilon : probabilité d'exploration
        max_steps : nombre total de steps réels
        N       : nombre de simulations par step réel
        cumulated_rewards : liste pour tracker les rewards

    Returns:
        Q  : action-value function apprise
        pi : policy optimale extraite de Q
    """

    # Initialisation de Q
    Q = np.zeros((env.max_state_count(), env.max_actions_count()))

    # Modèle : dictionnaire (s, a) → (r, s')
    model = {}

    total_steps = 0
    total_rewards = 0.0

    while total_steps < max_steps:
        env.reset()

        while not env.is_game_over() and total_steps < max_steps:
            s = env.current_state()

            # 1. Choisir une action epsilon-greedy
            a = choose_action_epsilon_greedy(Q, s, epsilon, env.available_actions())

            # 2. Jouer l'action réelle
            prev_score = env.score()
            env.step(a)
            total_steps += 1

            r = env.score() - prev_score
            total_rewards += r
            s_next = env.current_state()

            if cumulated_rewards is not None:
                cumulated_rewards.append(total_rewards)

            # 3. Mise à jour Q (Q-Learning)
            if env.is_game_over():
                Q[s, a] += alpha * (r - Q[s, a])
            else:
                Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])

            # 4. Sauvegarder dans le modèle
            model[(s, a)] = (r, s_next)

            # 5. N simulations (planification)
            for _ in range(N):
                # Tirer (s, a) au hasard depuis le modèle
                s_sim, a_sim = list(model.keys())[
                    np.random.randint(0, len(model))
                ]
                r_sim, s_next_sim = model[(s_sim, a_sim)]

                # Mise à jour Q avec l'expérience simulée
                Q[s_sim, a_sim] += alpha * (
                    r_sim + gamma * np.max(Q[s_next_sim]) - Q[s_sim, a_sim]
                )

    # Extraire la policy optimale depuis Q
    pi = np.zeros((env.max_state_count(), env.max_actions_count()))
    for s in range(env.max_state_count()):
        best_a = np.argmax(Q[s])
        pi[s, best_a] = 1.0

    return Q, pi

def dyna_q_plus(
    env,
    gamma: float = 0.999,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    max_steps: int = 10_000,
    N: int = 10,
    kappa: float = 0.001,
    cumulated_rewards: List[float] = None
):
    """
    Algorithme Dyna-Q+.

    Comme Dyna-Q mais ajoute un bonus de curiosité
    pour les expériences anciennes non visitées.

    Args:
        env     : environnement (ModelFreeEnv)
        gamma   : facteur de discount
        alpha   : taux d'apprentissage
        epsilon : probabilité d'exploration
        max_steps : nombre total de steps réels
        N       : nombre de simulations par step réel
        kappa   : bonus de curiosité (0 = Dyna-Q classique)
        cumulated_rewards : liste pour tracker les rewards

    Returns:
        Q  : action-value function apprise
        pi : policy optimale extraite de Q
    """

    # Initialisation de Q
    Q = np.zeros((env.max_state_count(), env.max_actions_count()))

    # Modèle : (s, a) → (r, s', last_time_visited)
    model = {}

    # Initialiser toutes les actions non visitées → (r=0, s=s, time=0)
    for s in range(env.max_state_count()):
        for a in range(env.max_actions_count()):
            model[(s, a)] = (0.0, s, 0)  # r=0, s'=s, last_time=0

    total_steps = 0
    total_rewards = 0.0

    while total_steps < max_steps:
        env.reset()

        while not env.is_game_over() and total_steps < max_steps:
            s = env.current_state()

            # 1. Choisir une action epsilon-greedy
            a = choose_action_epsilon_greedy(Q, s, epsilon, env.available_actions())

            # 2. Jouer l'action réelle
            prev_score = env.score()
            env.step(a)
            total_steps += 1

            r = env.score() - prev_score
            total_rewards += r
            s_next = env.current_state()

            if cumulated_rewards is not None:
                cumulated_rewards.append(total_rewards)

            # 3. Mise à jour Q (Q-Learning)
            if env.is_game_over():
                Q[s, a] += alpha * (r - Q[s, a])
            else:
                Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])

            # 4. Sauvegarder dans le modèle avec le temps actuel
            model[(s, a)] = (r, s_next, total_steps)

            # 5. N simulations avec bonus de curiosité
            for _ in range(N):
                # Tirer (s, a) au hasard depuis le modèle
                s_sim, a_sim = list(model.keys())[
                    np.random.randint(0, len(model))
                ]
                r_sim, s_next_sim, last_time = model[(s_sim, a_sim)]

                # Bonus de curiosité → favorise les expériences anciennes
                time_since_last = total_steps - last_time
                bonus = kappa * np.sqrt(time_since_last)

                # Mise à jour Q avec bonus
                Q[s_sim, a_sim] += alpha * (
                    r_sim + bonus + gamma * np.max(Q[s_next_sim]) - Q[s_sim, a_sim]
                )

    # Extraire la policy optimale depuis Q
    pi = np.zeros((env.max_state_count(), env.max_actions_count()))
    for s in range(env.max_state_count()):
        best_a = np.argmax(Q[s])
        pi[s, best_a] = 1.0

    return Q, pi