from grasp_env.grasp_env import GraspEnv

try:
    import gymnasium as gym
    gym.register(
        id="GraspEnv-v0",
        entry_point="grasp_env.grasp_env:GraspEnv",
        max_episode_steps=500,
    )
except Exception:
    pass  # gym registration is optional; import GraspEnv directly if preferred

__all__ = ["GraspEnv"]
