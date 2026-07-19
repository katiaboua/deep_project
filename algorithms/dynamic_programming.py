import numpy as np


def iterative_policy_evaluation(
    model_S: np.ndarray,
    model_A: np.ndarray,
    model_R: np.ndarray,
    model_p: np.ndarray,
    model_T: np.ndarray,
    pi: np.ndarray,
    gamma: float = 0.99999,
    theta: float = 0.00001
) -> np.ndarray:
    """
    Évalue une policy donnée en calculant la fonction de valeur V.
    """
    V = np.random.random(len(model_S))
    V[model_T] = 0.0

    while True:
        delta = 0.0
        for s_index in range(len(model_S)):
            v = V[s_index]
            total = 0.0
            for a_index in range(len(model_A)):
                a_total = 0.0
                for s_p_index in range(len(model_S)):
                    for r_index in range(len(model_R)):
                        a_total += model_p[s_index, a_index, s_p_index, r_index] * (
                            model_R[r_index] + gamma * V[s_p_index]
                        )
                total += pi[s_index, a_index] * a_total
            V[s_index] = total
            delta = np.maximum(delta, np.abs(total - v))
        if delta < theta:
            return V


def policy_iteration(
    model_S: np.ndarray,
    model_A: np.ndarray,
    model_R: np.ndarray,
    model_p: np.ndarray,
    model_T: np.ndarray,
    gamma: float = 0.99999,
    theta: float = 0.00001
):
    """
    Trouve la policy optimale en alternant évaluation et amélioration.

    Args:
        model_S : états
        model_A : actions
        model_R : rewards possibles
        model_p : matrice de transition p[s, a, s', r]
        model_T : états terminaux
        gamma   : facteur de discount
        theta   : seuil de convergence

    Returns:
        pi : policy optimale, matrice (|S|, |A|)
        V  : fonction de valeur optimale, vecteur de taille |S|
    """
    # Initialisation aléatoire de V et pi
    V = np.random.random(len(model_S))
    V[model_T] = 0.0

    pi = np.zeros((len(model_S), len(model_A)))
    for s_index in range(len(model_S)):
        a_random_index = np.random.randint(0, len(model_A))
        pi[s_index, a_random_index] = 1.0

    while True:
        # Étape 1 : Policy Evaluation
        while True:
            delta = 0.0
            for s_index in range(len(model_S)):
                v = V[s_index]
                total = 0.0
                for a_index in range(len(model_A)):
                    a_total = 0.0
                    for s_p_index in range(len(model_S)):
                        for r_index in range(len(model_R)):
                            a_total += model_p[s_index, a_index, s_p_index, r_index] * (
                                model_R[r_index] + gamma * V[s_p_index]
                            )
                    total += pi[s_index, a_index] * a_total
                V[s_index] = total
                delta = np.maximum(delta, np.abs(total - v))
            if delta < theta:
                break

        # Étape 2 : Policy Improvement
        policy_stable = True
        for s_index in range(len(model_S)):
            old_action_index = np.argmax(pi[s_index])

            best_a_index = None
            best_a_score = 0.0
            for a_index in range(len(model_A)):
                total = 0.0
                for s_p_index in range(len(model_S)):
                    for r_index in range(len(model_R)):
                        total += model_p[s_index, a_index, s_p_index, r_index] * (
                            model_R[r_index] + gamma * V[s_p_index]
                        )
                if best_a_index is None or total >= best_a_score:
                    best_a_index = a_index
                    best_a_score = total

            pi[s_index] = 0.0
            pi[s_index, best_a_index] = 1.0

            if old_action_index != best_a_index:
                policy_stable = False

        if policy_stable:
            return pi, V


def value_iteration(
    model_S: np.ndarray,
    model_A: np.ndarray,
    model_R: np.ndarray,
    model_p: np.ndarray,
    model_T: np.ndarray,
    gamma: float = 0.99999,
    theta: float = 0.00001
):
    """
    Trouve la policy optimale en calculant directement V optimal.

    Différence avec Policy Iteration :
        - Pas besoin d'évaluer une policy entièrement
        - On prend le MAX sur les actions directement
        - Plus rapide : une seule boucle au lieu de deux

    Args:
        model_S : états
        model_A : actions
        model_R : rewards possibles
        model_p : matrice de transition p[s, a, s', r]
        model_T : états terminaux
        gamma   : facteur de discount
        theta   : seuil de convergence

    Returns:
        pi : policy optimale, matrice (|S|, |A|)
        V  : fonction de valeur optimale, vecteur de taille |S|
    """
    # Initialisation aléatoire
    V = np.random.random(len(model_S))
    V[model_T] = 0.0

    while True:
        delta = 0.0
        for s_index in range(len(model_S)):
            v = V[s_index]

            # Pour chaque action, calculer la valeur
            best_value = None
            for a_index in range(len(model_A)):
                total = 0.0
                for s_p_index in range(len(model_S)):
                    for r_index in range(len(model_R)):
                        total += model_p[s_index, a_index, s_p_index, r_index] * (
                            model_R[r_index] + gamma * V[s_p_index]
                        )
                # On garde le MAX au lieu de suivre une policy
                if best_value is None or total > best_value:
                    best_value = total

            V[s_index] = best_value
            delta = np.maximum(delta, np.abs(best_value - v))

        if delta < theta:
            break

    # Extraire la policy à partir de V
    pi = np.zeros((len(model_S), len(model_A)))
    for s_index in range(len(model_S)):
        best_a_index = None
        best_a_score = None
        for a_index in range(len(model_A)):
            total = 0.0
            for s_p_index in range(len(model_S)):
                for r_index in range(len(model_R)):
                    total += model_p[s_index, a_index, s_p_index, r_index] * (
                        model_R[r_index] + gamma * V[s_p_index]
                    )
            if best_a_index is None or total > best_a_score:
                best_a_index = a_index
                best_a_score = total
        pi[s_index] = 0.0
        pi[s_index, best_a_index] = 1.0

    return pi, V
