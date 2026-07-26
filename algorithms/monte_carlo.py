import numpy as np


def monte_carlo_es(env, gamma: float = 0.999, num_episodes: int = 10000, max_steps: int = 100):
    """
    Monte Carlo Exploring Starts (ES) Control.

    Trouve la policy optimale en jouant des épisodes complets et en mettant
    à jour Q(s,a) par moyenne des retours observés (first-visit).

    Args:
        env         : environnement respectant le contrat ModelFreeEnv
        gamma       : facteur de discount
        num_episodes: nombre d'épisodes à jouer
        max_steps   : garde-fou pour éviter les épisodes infinis

    Returns:
        pi : policy optimale, matrice (nb_states, nb_actions)
        Q  : fonction action-valeur, matrice (nb_states, nb_actions)
    """
    nb_states = env.max_state_count()
    nb_actions = env.max_actions_count()

    Q = np.random.random((nb_states, nb_actions))
    returns_sum = np.zeros((nb_states, nb_actions))
    returns_count = np.zeros((nb_states, nb_actions))

    pi = np.zeros((nb_states, nb_actions), dtype=int)
    for s in range(nb_states):
        pi[s, np.random.randint(nb_actions)] = 1

    for _ in range(num_episodes):
        env.reset()

        # Exploring start : premier état/action forcés aléatoirement
        # (on ignore les actions indisponibles pour le vrai premier état si besoin)
        first_action = np.random.choice(env.available_actions())

        episode = []  # liste de (state, action, reward)
        action = first_action
        step_count = 0

        while not env.is_game_over() and step_count < max_steps:
            state = env.current_state()
            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score
            episode.append((state, action, reward))

            if not env.is_game_over():
                # argmax restreint aux actions disponibles (necessaire pour
                # les envs ou toutes les actions ne sont pas toujours dispos,
                # ex: Secret Env 2/3)
                next_state = env.current_state()
                available = env.available_actions()
                action = int(max(available, key=lambda a: pi[next_state, a]))
            step_count += 1

        # Calcul des retours G et mise à jour first-visit (en partant de la fin)
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                returns_sum[s, a] += G
                returns_count[s, a] += 1
                Q[s, a] = returns_sum[s, a] / returns_count[s, a]

                # Amélioration de la policy pour cet état
                best_a = np.argmax(Q[s])
                pi[s] = 0
                pi[s, best_a] = 1

    return pi, Q


def on_policy_first_visit_mc_control(
    env,
    gamma: float = 0.999,
    epsilon: float = 0.1,
    num_episodes: int = 10000,
    max_steps: int = 100
):
    """
    On-policy First-Visit Monte Carlo Control (epsilon-greedy).

    Contrairement à MC ES, l'exploration vient d'une policy epsilon-greedy
    jouée du début à la fin de l'épisode (pas besoin de démarrage forcé).

    Args:
        env         : environnement respectant le contrat ModelFreeEnv
        gamma       : facteur de discount
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
    returns_sum = np.zeros((nb_states, nb_actions))
    returns_count = np.zeros((nb_states, nb_actions))

    def epsilon_greedy_action(state):
        available = env.available_actions()
        if np.random.random() < epsilon:
            return np.random.choice(available)
        # argmax restreint aux actions disponibles
        q_values = Q[state]
        best_a = max(available, key=lambda a: q_values[a])
        return best_a

    for _ in range(num_episodes):
        env.reset()
        episode = []
        step_count = 0

        while not env.is_game_over() and step_count < max_steps:
            state = env.current_state()
            action = epsilon_greedy_action(state)
            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score
            episode.append((state, action, reward))
            step_count += 1

        # Mise à jour first-visit, en partant de la fin de l'épisode
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                returns_sum[s, a] += G
                returns_count[s, a] += 1
                Q[s, a] = returns_sum[s, a] / returns_count[s, a]

    # Policy finale : gloutonne pure (epsilon=0) à partir de Q appris
    pi = np.zeros((nb_states, nb_actions), dtype=int)
    for s in range(nb_states):
        pi[s, np.argmax(Q[s])] = 1

    return pi, Q


def off_policy_mc_control(
    env,
    gamma: float = 0.999,
    num_episodes: int = 10000,
    max_steps: int = 100
):
    """
    Off-policy Monte Carlo Control avec importance sampling pondéré (incrémental).

    - behavior policy b : aléatoire uniforme (génère les épisodes)
    - target policy pi  : gloutonne sur Q (celle qu'on optimise)

    Args:
        env         : environnement respectant le contrat ModelFreeEnv
        gamma       : facteur de discount
        num_episodes: nombre d'épisodes à jouer
        max_steps   : garde-fou anti-boucle infinie

    Returns:
        pi : policy optimale (déterministe, argmax de Q), matrice (nb_states, nb_actions)
        Q  : fonction action-valeur, matrice (nb_states, nb_actions)
    """
    nb_states = env.max_state_count()
    nb_actions = env.max_actions_count()

    Q = np.random.random((nb_states, nb_actions)) * 1e-3
    C = np.zeros((nb_states, nb_actions))  # somme cumulée des poids

    # Target policy déterministe, initialisée arbitrairement (argmax de Q)
    pi = np.zeros(nb_states, dtype=int)
    for s in range(nb_states):
        pi[s] = np.argmax(Q[s])

    for _ in range(num_episodes):
        env.reset()
        episode = []
        step_count = 0

        # Génération de l'épisode avec la behavior policy (aléatoire uniforme)
        while not env.is_game_over() and step_count < max_steps:
            state = env.current_state()
            available = env.available_actions()
            action = np.random.choice(available)
            b_prob = 1.0 / len(available)

            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score

            episode.append((state, action, reward, b_prob))
            step_count += 1

        G = 0.0
        W = 1.0
        for t in reversed(range(len(episode))):
            s, a, r, b_prob = episode[t]
            G = gamma * G + r

            C[s, a] += W
            Q[s, a] += (W / C[s, a]) * (G - Q[s, a])

            # Mise à jour de la target policy pour cet état
            pi[s] = np.argmax(Q[s])

            # Si l'action jouée par b n'est pas celle que pi aurait choisie,
            # on arrête la mise à jour pour ce reste d'épisode (poids=0 sinon)
            if a != pi[s]:
                break

            W = W * (1.0 / b_prob)

    # Conversion pi en format one-hot pour cohérence avec les autres algos
    pi_onehot = np.zeros((nb_states, nb_actions))
    for s in range(nb_states):
        pi_onehot[s, pi[s]] = 1

    return pi_onehot, Q