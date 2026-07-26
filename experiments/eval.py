import numpy as np


def evaluate_policy(env, pi: np.ndarray, num_episodes: int = 1000, max_steps: int = 100) -> dict:
    scores = []
    for _ in range(num_episodes):
        env.reset()
        steps = 0
        while not env.is_game_over() and steps < max_steps:
            available = env.available_actions()
            preferred_action = np.argmax(pi[env.current_state()])
            if preferred_action in available:
                action = preferred_action
            else:
                # L'etat n'a jamais ete visite pendant l'entrainement (policy par defaut invalide ici)
                # -> on prend une action valide de secours pour ne pas planter
                action = available[0]
            env.step(action)
            steps += 1
        scores.append(env.score())

    scores = np.array(scores)
    return {
        "mean_score": round(float(scores.mean()), 4),
        "std_score": round(float(scores.std()), 4),
        "win_rate": round(float((scores > 0).mean()), 4),
    }