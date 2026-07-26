import numpy as np


def sarsa(env, gamma: float = 0.999, alpha: float = 0.1, epsilon: float = 0.1,
          num_episodes: int = 10000, max_steps: int = 100, episode_scores: list = None):
    nb_states = env.max_state_count()
    nb_actions = env.max_actions_count()
    Q = np.zeros((nb_states, nb_actions))
    valid_actions_by_state = {}

    def epsilon_greedy_action(state, available):
        if np.random.random() < epsilon:
            return np.random.choice(available)
        q_values = Q[state]
        return max(available, key=lambda a: q_values[a])

    for _ in range(num_episodes):
        env.reset()
        state = env.current_state()
        available = env.available_actions()
        valid_actions_by_state[state] = available
        action = epsilon_greedy_action(state, available)
        step_count = 0

        while not env.is_game_over() and step_count < max_steps:
            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score

            if env.is_game_over():
                Q[state, action] += alpha * (reward - Q[state, action])
                break

            next_state = env.current_state()
            next_available = env.available_actions()
            valid_actions_by_state[next_state] = next_available
            next_action = epsilon_greedy_action(next_state, next_available)

            Q[state, action] += alpha * (
                reward + gamma * Q[next_state, next_action] - Q[state, action]
            )
            state = next_state
            action = next_action
            step_count += 1

        if episode_scores is not None:
            episode_scores.append(env.score())

    pi = np.zeros((nb_states, nb_actions), dtype=int)
    for s in range(nb_states):
        actions_here = valid_actions_by_state.get(s)
        best_a = max(actions_here, key=lambda a: Q[s, a]) if actions_here is not None else int(np.argmax(Q[s]))
        pi[s, best_a] = 1

    return pi, Q


def q_learning(env, gamma: float = 0.999, alpha: float = 0.1, epsilon: float = 0.1,
               num_episodes: int = 10000, max_steps: int = 100, episode_scores: list = None):
    nb_states = env.max_state_count()
    nb_actions = env.max_actions_count()
    Q = np.zeros((nb_states, nb_actions))
    valid_actions_by_state = {}

    def epsilon_greedy_action(state, available):
        if np.random.random() < epsilon:
            return np.random.choice(available)
        q_values = Q[state]
        return max(available, key=lambda a: q_values[a])

    for _ in range(num_episodes):
        env.reset()
        step_count = 0

        while not env.is_game_over() and step_count < max_steps:
            state = env.current_state()
            available = env.available_actions()
            valid_actions_by_state[state] = available
            action = epsilon_greedy_action(state, available)

            prev_score = env.score()
            env.step(action)
            reward = env.score() - prev_score

            if env.is_game_over():
                Q[state, action] += alpha * (reward - Q[state, action])
            else:
                next_state = env.current_state()
                next_available = env.available_actions()
                max_q_next = max(Q[next_state, a] for a in next_available)
                Q[state, action] += alpha * (
                    reward + gamma * max_q_next - Q[state, action]
                )
            step_count += 1

        if episode_scores is not None:
            episode_scores.append(env.score())

    pi = np.zeros((nb_states, nb_actions), dtype=int)
    for s in range(nb_states):
        actions_here = valid_actions_by_state.get(s)
        best_a = max(actions_here, key=lambda a: Q[s, a]) if actions_here is not None else int(np.argmax(Q[s]))
        pi[s, best_a] = 1

    return pi, Q