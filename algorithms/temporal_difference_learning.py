import numpy as np


def sarsa(
    env,
    gamma: float = 0.999,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    num_episodes: int = 10000,
    max_steps: int = 100
):
    """
    Sarsa (on-policy TD Control).

    Met à jour Q(s,a) à chaque pas via la transition (s, a, r, s', a'),
    avec a et a' choisis selon la même policy epsilon-greedy.

    Args:
        env         : environnement respectant le contrat ModelFreeEnv
        gamma       : facteur de discount
        alpha       : taux d'apprentissage
        epsilon     : probabilité de choisir une action aléatoire
        num_episodes: nombre d'épisodes à jouer
        max_steps   : garde-fou anti-boucle infinie

    Returns:
        pi : policy optimale (déterministe, argmax de Q), matrice (nb_states, nb_actions)
        Q  : fonction action-valeur, matrice (nb_states, nb_actions)
    """
    nb_states = env.max_state_count()
    nb_actions = env.max_actions_count()

    Q = np.zeros((nb_states, nb_actions))

    def epsilon_greedy_action(state):
        available = env.available_actions()
        if np.random.random() < epsilon:
            return np.random.choice(available)
        q_values = Q[state]
        return max(available, key=lambda a: q_values[a])

    for _ in range(num_episodes):
        env.reset()
        state = env.current_state()
        action = epsilon_greedy_action(state)
        step_count = 0

        while not env.is_game_over() and step_count < max_steps:
            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score

            if env.is_game_over():
                # Pas de state' ni action' : Q(terminal) = 0
                Q[state, action] += alpha * (reward - Q[state, action])
                break

            next_state = env.current_state()
            next_action = epsilon_greedy_action(next_state)

            # Mise à jour Sarsa : utilise l'action RÉELLEMENT choisie pour next_state
            Q[state, action] += alpha * (
                reward + gamma * Q[next_state, next_action] - Q[state, action]
            )

            state = next_state
            action = next_action
            step_count += 1

    # Policy finale : gloutonne pure à partir de Q appris
    pi = np.zeros((nb_states, nb_actions), dtype=int)
    for s in range(nb_states):
        pi[s, np.argmax(Q[s])] = 1

    return pi, Q
def q_learning(
    env,
    gamma: float = 0.999,
    alpha: float = 0.1,
    epsilon: float = 0.1,
    num_episodes: int = 10000,
    max_steps: int = 100
):
    """
    Q-Learning (off-policy TD Control).

    Met à jour Q(s,a) à chaque pas en utilisant le MAX de Q(s', .),
    indépendamment de l'action réellement jouée à l'état suivant.

    Args:
        env         : environnement respectant le contrat ModelFreeEnv
        gamma       : facteur de discount
        alpha       : taux d'apprentissage
        epsilon     : probabilité de choisir une action aléatoire (exploration)
        num_episodes: nombre d'épisodes à jouer
        max_steps   : garde-fou anti-boucle infinie

    Returns:
        pi : policy optimale (déterministe, argmax de Q), matrice (nb_states, nb_actions)
        Q  : fonction action-valeur, matrice (nb_states, nb_actions)
    """
    nb_states = env.max_state_count()
    nb_actions = env.max_actions_count()

    Q = np.zeros((nb_states, nb_actions))

    def epsilon_greedy_action(state):
        available = env.available_actions()
        if np.random.random() < epsilon:
            return np.random.choice(available)
        q_values = Q[state]
        return max(available, key=lambda a: q_values[a])

    for _ in range(num_episodes):
        env.reset()
        step_count = 0

        while not env.is_game_over() and step_count < max_steps:
            state = env.current_state()
            action = epsilon_greedy_action(state)

            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score

            if env.is_game_over():
                Q[state, action] += alpha * (reward - Q[state, action])
            else:
                next_state = env.current_state()
                # Différence clé avec Sarsa : max sur les actions dispo, pas l'action réelle
                next_available = env.available_actions()
                max_q_next = max(Q[next_state, a] for a in next_available)
                Q[state, action] += alpha * (
                    reward + gamma * max_q_next - Q[state, action]
                )

            step_count += 1

    # Policy finale : gloutonne pure à partir de Q appris
    pi = np.zeros((nb_states, nb_actions), dtype=int)
    for s in range(nb_states):
        pi[s, np.argmax(Q[s])] = 1

    return pi, Q