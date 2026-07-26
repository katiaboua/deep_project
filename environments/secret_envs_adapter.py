import numpy as np
from environments.secret_envs import SecretEnv0, SecretEnv1, SecretEnv2, SecretEnv3


class SecretEnvAdapter:
    """
    Adaptateur pour brancher les environnements secrets
    sur nos algorithmes RL (qui utilisent le contrat ModelFreeEnv).
    """

    def __init__(self, secret_env):
        self.env = secret_env

    def reset(self):
        self.env.reset()

    def step(self, action: int):
        self.env.step(action)

    def is_game_over(self) -> bool:
        return self.env.is_game_over()

    def current_state(self) -> int:
        return self.env.state_id()

    def available_actions(self):
        return self.env.available_actions()

    def score(self) -> float:
        return self.env.score()

    def max_state_count(self) -> int:
        return self.env.num_states()

    def max_actions_count(self) -> int:
        return self.env.num_actions()

    def pretty_print(self):
        self.env.display()


class SecretEnv0Adapted(SecretEnvAdapter):
    def __init__(self):
        super().__init__(SecretEnv0())

class SecretEnv1Adapted(SecretEnvAdapter):
    def __init__(self):
        super().__init__(SecretEnv1())

class SecretEnv2Adapted(SecretEnvAdapter):
    def __init__(self):
        super().__init__(SecretEnv2())

class SecretEnv3Adapted(SecretEnvAdapter):
    def __init__(self):
        super().__init__(SecretEnv3())