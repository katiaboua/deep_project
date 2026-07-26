import ctypes
import os
import platform

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIBS_DIR = os.path.join(_HERE, "libs")

if platform.system().lower() == "windows":
    lib_path = os.path.join(_LIBS_DIR, "secret_envs.dll")
elif platform.system().lower() == "linux":
    lib_path = os.path.join(_LIBS_DIR, "libsecret_envs.so")
elif platform.system().lower() == "darwin":
    if "intel" in platform.processor().lower():
        lib_path = os.path.join(_LIBS_DIR, "libsecret_envs_intel_macos.dylib")
    else:
        lib_path = os.path.join(_LIBS_DIR, "libsecret_envs.dylib")


class SecretEnv2Wrapper:
    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary(lib_path)

        self.lib.secret_env_2_num_states.argtypes = []
        self.lib.secret_env_2_num_states.restype = ctypes.c_size_t

        self.lib.secret_env_2_num_actions.argtypes = []
        self.lib.secret_env_2_num_actions.restype = ctypes.c_size_t

        self.lib.secret_env_2_num_rewards.argtypes = []
        self.lib.secret_env_2_num_rewards.restype = ctypes.c_size_t

        self.lib.secret_env_2_reward.argtypes = [ctypes.c_size_t]
        self.lib.secret_env_2_reward.restype = ctypes.c_float

        self.lib.secret_env_2_transition_probability.argtypes = [
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t
        ]
        self.lib.secret_env_2_transition_probability.restype = ctypes.c_float

        self.lib.secret_env_2_new.argtypes = []
        self.lib.secret_env_2_new.restype = ctypes.c_void_p

        self.lib.secret_env_2_reset.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_2_reset.restype = None

        self.lib.secret_env_2_display.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_2_display.restype = None

        self.lib.secret_env_2_state_id.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_2_state_id.restype = ctypes.c_size_t

        self.lib.secret_env_2_is_forbidden.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.secret_env_2_is_forbidden.restype = ctypes.c_bool

        self.lib.secret_env_2_is_game_over.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_2_is_game_over.restype = ctypes.c_bool

        self.lib.secret_env_2_available_actions.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_2_available_actions.restype = ctypes.POINTER(ctypes.c_size_t)

        self.lib.secret_env_2_available_actions_len.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_2_available_actions_len.restype = ctypes.c_size_t

        self.lib.secret_env_2_available_actions_delete.argtypes = [
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t
        ]
        self.lib.secret_env_2_available_actions_delete.restype = None

        self.lib.secret_env_2_step.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.secret_env_2_step.restype = None

        self.lib.secret_env_2_score.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_2_score.restype = ctypes.c_float

        self.lib.secret_env_2_delete.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_2_delete.restype = None

        self.lib.secret_env_2_from_random_state.argtypes = []
        self.lib.secret_env_2_from_random_state.restype = ctypes.c_void_p


class SecretEnv2Raw:
    def __init__(self, wrapper=None, instance=None):
        if wrapper is None:
            wrapper = SecretEnv2Wrapper()
        self.wrapper = wrapper
        if instance is None:
            instance = self.wrapper.lib.secret_env_2_new()
        self.instance = instance

    def __del__(self):
        if self.wrapper is not None and self.instance is not None:
            try:
                self.wrapper.lib.secret_env_2_delete(self.instance)
            except Exception:
                pass

    def num_states(self) -> int:
        return self.wrapper.lib.secret_env_2_num_states()

    def num_actions(self) -> int:
        return self.wrapper.lib.secret_env_2_num_actions()

    def num_rewards(self) -> int:
        return self.wrapper.lib.secret_env_2_num_rewards()

    def reward(self, i: int) -> float:
        return self.wrapper.lib.secret_env_2_reward(i)

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        return self.wrapper.lib.secret_env_2_transition_probability(s, a, s_p, r_index)

    def state_id(self) -> int:
        return self.wrapper.lib.secret_env_2_state_id(self.instance)

    def reset(self):
        self.wrapper.lib.secret_env_2_reset(self.instance)

    def display(self):
        self.wrapper.lib.secret_env_2_display(self.instance)

    def is_forbidden(self, action: int) -> int:
        return self.wrapper.lib.secret_env_2_is_forbidden(self.instance, action)

    def is_game_over(self) -> bool:
        return self.wrapper.lib.secret_env_2_is_game_over(self.instance)

    def available_actions(self) -> np.ndarray:
        actions_len = self.wrapper.lib.secret_env_2_available_actions_len(self.instance)
        actions_pointer = self.wrapper.lib.secret_env_2_available_actions(self.instance)
        arr = np.ctypeslib.as_array(actions_pointer, (actions_len,))
        arr_copy = np.copy(arr)
        self.wrapper.lib.secret_env_2_available_actions_delete(actions_pointer, actions_len)
        return arr_copy

    def step(self, action: int):
        self.wrapper.lib.secret_env_2_step(self.instance, action)

    def score(self):
        return self.wrapper.lib.secret_env_2_score(self.instance)


class SecretEnv3Wrapper:
    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary(lib_path)

        self.lib.secret_env_3_num_states.argtypes = []
        self.lib.secret_env_3_num_states.restype = ctypes.c_size_t

        self.lib.secret_env_3_num_actions.argtypes = []
        self.lib.secret_env_3_num_actions.restype = ctypes.c_size_t

        self.lib.secret_env_3_num_rewards.argtypes = []
        self.lib.secret_env_3_num_rewards.restype = ctypes.c_size_t

        self.lib.secret_env_3_reward.argtypes = [ctypes.c_size_t]
        self.lib.secret_env_3_reward.restype = ctypes.c_float

        self.lib.secret_env_3_transition_probability.argtypes = [
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t
        ]
        self.lib.secret_env_3_transition_probability.restype = ctypes.c_float

        self.lib.secret_env_3_new.argtypes = []
        self.lib.secret_env_3_new.restype = ctypes.c_void_p

        self.lib.secret_env_3_reset.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_3_reset.restype = None

        self.lib.secret_env_3_display.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_3_display.restype = None

        self.lib.secret_env_3_state_id.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_3_state_id.restype = ctypes.c_size_t

        self.lib.secret_env_3_is_forbidden.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.secret_env_3_is_forbidden.restype = ctypes.c_bool

        self.lib.secret_env_3_is_game_over.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_3_is_game_over.restype = ctypes.c_bool

        self.lib.secret_env_3_available_actions.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_3_available_actions.restype = ctypes.POINTER(ctypes.c_size_t)

        self.lib.secret_env_3_available_actions_len.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_3_available_actions_len.restype = ctypes.c_size_t

        self.lib.secret_env_3_available_actions_delete.argtypes = [
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t
        ]
        self.lib.secret_env_3_available_actions_delete.restype = None

        self.lib.secret_env_3_step.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.secret_env_3_step.restype = None

        self.lib.secret_env_3_score.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_3_score.restype = ctypes.c_float

        self.lib.secret_env_3_delete.argtypes = [ctypes.c_void_p]
        self.lib.secret_env_3_delete.restype = None

        self.lib.secret_env_3_from_random_state.argtypes = []
        self.lib.secret_env_3_from_random_state.restype = ctypes.c_void_p


class SecretEnv3Raw:
    def __init__(self, wrapper=None, instance=None):
        if wrapper is None:
            wrapper = SecretEnv3Wrapper()
        self.wrapper = wrapper
        if instance is None:
            instance = self.wrapper.lib.secret_env_3_new()
        self.instance = instance

    def __del__(self):
        if self.wrapper is not None and self.instance is not None:
            try:
                self.wrapper.lib.secret_env_3_delete(self.instance)
            except Exception:
                pass

    def num_states(self) -> int:
        return self.wrapper.lib.secret_env_3_num_states()

    def num_actions(self) -> int:
        return self.wrapper.lib.secret_env_3_num_actions()

    def num_rewards(self) -> int:
        return self.wrapper.lib.secret_env_3_num_rewards()

    def reward(self, i: int) -> float:
        return self.wrapper.lib.secret_env_3_reward(i)

    def p(self, s: int, a: int, s_p: int, r_index: int) -> float:
        return self.wrapper.lib.secret_env_3_transition_probability(s, a, s_p, r_index)

    def state_id(self) -> int:
        return self.wrapper.lib.secret_env_3_state_id(self.instance)

    def reset(self):
        self.wrapper.lib.secret_env_3_reset(self.instance)

    def display(self):
        self.wrapper.lib.secret_env_3_display(self.instance)

    def is_forbidden(self, action: int) -> int:
        return self.wrapper.lib.secret_env_3_is_forbidden(self.instance, action)

    def is_game_over(self) -> bool:
        return self.wrapper.lib.secret_env_3_is_game_over(self.instance)

    def available_actions(self) -> np.ndarray:
        actions_len = self.wrapper.lib.secret_env_3_available_actions_len(self.instance)
        actions_pointer = self.wrapper.lib.secret_env_3_available_actions(self.instance)
        arr = np.ctypeslib.as_array(actions_pointer, (actions_len,))
        arr_copy = np.copy(arr)
        self.wrapper.lib.secret_env_3_available_actions_delete(actions_pointer, actions_len)
        return arr_copy

    def step(self, action: int):
        self.wrapper.lib.secret_env_3_step(self.instance, action)

    def score(self):
        return self.wrapper.lib.secret_env_3_score(self.instance)
