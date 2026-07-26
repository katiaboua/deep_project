import numpy as np


def monte_carlo_es(env, gamma: float = 0.999, num_episodes: int = 10000, max_steps: int = 100):
    """
    Monte Carlo Exploring Starts (ES) Control.

    Les actions sont toujours choisies parmi env.available_actions(),
    jamais sur l'ensemble des nb_actions (certains états ont moins d'actions valides,
    et jouer une action interdite fait planter certains environnements).
    """
    nb_states = env.max_state_count()
    nb_actions = env.max_actions_count()

    Q = np.zeros((nb_states, nb_actions))
    returns_sum = np.zeros((nb_states, nb_actions))
    returns_count = np.zeros((nb_states, nb_actions))

    def greedy_action(state, available):
        q_values = Q[state]
        return max(available, key=lambda a: q_values[a])

    for _ in range(num_episodes):
        env.reset()
        action = np.random.choice(env.available_actions())

        episode = []
        step_count = 0

        while not env.is_game_over() and step_count < max_steps:
            state = env.current_state()
            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score
            episode.append((state, action, reward))

            if not env.is_game_over():
                action = greedy_action(env.current_state(), env.available_actions())
            step_count += 1

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

    pi = np.zeros((nb_states, nb_actions), dtype=int)
    for s in range(nb_states):
        seen_actions = [a for a in range(nb_actions) if returns_count[s, a] > 0]
        if seen_actions:
            best_a = max(seen_actions, key=lambda a: Q[s, a])
        else:
            best_a = int(np.argmax(Q[s]))
        pi[s, best_a] = 1

    return pi, Q


def on_policy_first_visit_mc_control(env, gamma: float = 0.999, epsilon: float = 0.1,
                                      num_episodes: int = 10000, max_steps: int = 100):
    nb_states = env.max_state_count()
    nb_actions = env.max_actions_count()

    Q = np.zeros((nb_states, nb_actions))
    returns_sum = np.zeros((nb_states, nb_actions))
    returns_count = np.zeros((nb_states, nb_actions))
    valid_actions_by_state = {}

    def epsilon_greedy_action(state, available):
        if np.random.random() < epsilon:
            return np.random.choice(available)
        q_values = Q[state]
        return max(available, key=lambda a: q_values[a])

    for _ in range(num_episodes):
        env.reset()
        episode = []
        step_count = 0
        while not env.is_game_over() and step_count < max_steps:
            state = env.current_state()
            available = env.available_actions()
            valid_actions_by_state[state] = available
            action = epsilon_greedy_action(state, available)
            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score
            episode.append((state, action, reward))
            step_count += 1

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

    pi = np.zeros((nb_states, nb_actions), dtype=int)
    for s in range(nb_states):
        actions_here = valid_actions_by_state.get(s)
        best_a = max(actions_here, key=lambda a: Q[s, a]) if actions_here is not None else int(np.argmax(Q[s]))
        pi[s, best_a] = 1

    return pi, Q


def off_policy_mc_control(env, gamma: float = 0.999, num_episodes: int = 10000, max_steps: int = 100):
    nb_states = env.max_state_count()
    nb_actions = env.max_actions_count()

    Q = np.zeros((nb_states, nb_actions))
    C = np.zeros((nb_states, nb_actions))
    pi = np.zeros(nb_states, dtype=int)

    for _ in range(num_episodes):
        env.reset()
        episode = []
        step_count = 0
        while not env.is_game_over() and step_count < max_steps:
            state = env.current_state()
            available = env.available_actions()
            action = np.random.choice(available)
            b_prob = 1.0 / len(available)
            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score
            episode.append((state, action, reward, b_prob, available))
            step_count += 1

        G = 0.0
        W = 1.0
        for t in reversed(range(len(episode))):
            s, a, r, b_prob, available = episode[t]
            G = gamma * G + r
            C[s, a] += W
            Q[s, a] += (W / C[s, a]) * (G - Q[s, a])
            pi[s] = max(available, key=lambda x: Q[s, x])
            if a != pi[s]:
                break
            W = W * (1.0 / b_prob)

    pi_onehot = np.zeros((nb_states, nb_actions))
    for s in range(nb_states):
        pi_onehot[s, pi[s]] = 1

    return pi_onehot, Q