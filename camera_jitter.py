import gymnasium as gym
import gymnasium_robotics  # noqa: F401  registers FrankaKitchen-v1
import mujoco
import numpy as np
from gymnasium_robotics.envs.franka_kitchen.franka_env import DEFAULT_CAMERA_CONFIG

import frames


class CameraJitter:
    """Re-renders a demo from its stored qpos (exact replay) with the scene camera moved, one pose per episode."""

    def __init__(self, image_size=frames.VLM_SIZE, render_size=frames.RENDER_SIZE,
                 azimuth=20.0, elevation=8.0, distance=0.3, lookat=0.06, seed=0):
        self.image_size = image_size
        # Uniform half-widths around the recorded pose: degrees, degrees, metres, metres per lookat axis.
        self.azimuth = azimuth
        self.elevation = elevation
        self.distance = distance
        self.lookat = lookat
        self.rng = np.random.default_rng(seed)
        self.env = gym.make("FrankaKitchen-v1", tasks_to_complete=["kettle"], render_mode="rgb_array")
        robot = self.env.unwrapped.robot_env
        self.renderer = robot.mujoco_renderer
        self.model, self.data = robot.model, robot.data
        robot.width = robot.height = self.renderer.width = self.renderer.height = render_size
        self.model.vis.global_.offwidth = self.model.vis.global_.offheight = render_size
        self.env.reset(seed=0)

    def sample_pose(self):
        """One free-camera config drawn around the recorded pose."""
        pose = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in DEFAULT_CAMERA_CONFIG.items()}
        pose["azimuth"] += self.rng.uniform(-self.azimuth, self.azimuth)
        pose["elevation"] += self.rng.uniform(-self.elevation, self.elevation)
        pose["distance"] += self.rng.uniform(-self.distance, self.distance)
        pose["lookat"] = pose["lookat"] + self.rng.uniform(-self.lookat, self.lookat, 3)
        return pose

    def render_episode(self, qpos, pose=None):
        """Frames for every step of qpos (T, nq) at image_size, from pose or a freshly sampled one."""
        self.renderer.default_cam_config = pose if pose is not None else self.sample_pose()
        self.renderer.camera_id = -1
        out = np.empty((len(qpos), self.image_size, self.image_size, 3), dtype=np.uint8)
        for t, q in enumerate(qpos):
            self.data.qpos[:] = q[:self.model.nq]
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)
            if t == 0:
                # The viewer only reads default_cam_config when it's created, so apply it to an existing one by hand.
                self.renderer._get_viewer("rgb_array")
                self.renderer._set_cam_config()
            out[t] = frames.resize(self.renderer.render("rgb_array"), self.image_size)
        return out

    def close(self):
        self.env.close()
