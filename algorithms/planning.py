import numpy as np
from typing import List


def choose_action_epsilon_greedy(Q: np.ndarray, s: int, epsilon: float, available_actions: List[int]) -> int:
    if np.random.random() < epsilon:
        return np.random.choice(available_actions)
    q_values = Q[s]
    return max(available_actions, key=lambda a: q_values[a])


def dyna_q(env, gamma: float = 0.999, alpha: float = 0.1, epsilon: float = 0.1,
           max_steps: int = 10_000, N: int = 10, cumulated_rewards: List[float] = None):
    Q = np.zeros((env.max_state_count(), env.max_actions_count()))
    model = {}
    valid_actions_by_state = {}

    total_steps = 0
    total_rewards = 0.0

    while total_steps < max_steps:
        env.reset()
        while not env.is_game_over() and total_steps < max_steps:
            s = env.current_state()
            available = env.available_actions()
            valid_actions_by_state[s] = available

            a = choose_action_epsilon_greedy(Q, s, epsilon, available)

            prev_score = env.score()
            env.step(a)
            total_steps += 1
            r = env.score() - prev_score
            total_rewards += r
            s_next = env.current_state()

            if cumulated_rewards is not None:
                cumulated_rewards.append(total_rewards)

            if env.is_game_over():
                Q[s, a] += alpha * (r - Q[s, a])
            else:
                Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])

            model[(s, a)] = (r, s_next)

            for _ in range(N):
                s_sim, a_sim = list(model.keys())[np.random.randint(0, len(model))]
                r_sim, s_next_sim = model[(s_sim, a_sim)]
                Q[s_sim, a_sim] += alpha * (r_sim + gamma * np.max(Q[s_next_sim]) - Q[s_sim, a_sim])

    pi = np.zeros((env.max_state_count(), env.max_actions_count()))
    for s in range(env.max_state_count()):
        actions_here = valid_actions_by_state.get(s)
        best_a = max(actions_here, key=lambda a: Q[s, a]) if actions_here is not None else int(np.argmax(Q[s]))
        pi[s, best_a] = 1.0

    return Q, pi


def dyna_q_plus(env, gamma: float = 0.999, alpha: float = 0.1, epsilon: float = 0.1,
                 max_steps: int = 10_000, N: int = 10, kappa: float = 0.001,
                 cumulated_rewards: List[float] = None):
    Q = np.zeros((env.max_state_count(), env.max_actions_count()))
    model = {}
    valid_actions_by_state = {}

    total_steps = 0
    total_rewards = 0.0

    while total_steps < max_steps:
        env.reset()
        while not env.is_game_over() and total_steps < max_steps:
            s = env.current_state()
            available = env.available_actions()
            valid_actions_by_state[s] = available

            for a_valid in available:
                if (s, a_valid) not in model:
                    model[(s, a_valid)] = (0.0, s, 0)

            a = choose_action_epsilon_greedy(Q, s, epsilon, available)

            prev_score = env.score()
            env.step(a)
            total_steps += 1
            r = env.score() - prev_score
            total_rewards += r
            s_next = env.current_state()

            if cumulated_rewards is not None:
                cumulated_rewards.append(total_rewards)

            if env.is_game_over():
                Q[s, a] += alpha * (r - Q[s, a])
            else:
                Q[s, a] += alpha * (r + gamma * np.max(Q[s_next]) - Q[s, a])

            model[(s, a)] = (r, s_next, total_steps)

            for _ in range(N):
                s_sim, a_sim = list(model.keys())[np.random.randint(0, len(model))]
                r_sim, s_next_sim, last_time = model[(s_sim, a_sim)]
                bonus = kappa * np.sqrt(total_steps - last_time)
                Q[s_sim, a_sim] += alpha * (r_sim + bonus + gamma * np.max(Q[s_next_sim]) - Q[s_sim, a_sim])

    pi = np.zeros((env.max_state_count(), env.max_actions_count()))
    for s in range(env.max_state_count()):
        actions_here = valid_actions_by_state.get(s)
        best_a = max(actions_here, key=lambda a: Q[s, a]) if actions_here is not None else int(np.argmax(Q[s]))
        pi[s, best_a] = 1.0

    return Q, pi