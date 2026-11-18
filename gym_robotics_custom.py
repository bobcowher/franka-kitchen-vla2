import time
import gymnasium as gym
import gymnasium_robotics  # registers FrankaKitchen-v1; no longer automatic in gymnasium 1.x
import mujoco
import numpy as np
from gymnasium import ObservationWrapper, Wrapper, spaces
from gymnasium_robotics.envs.franka_kitchen.kitchen_env import OBS_ELEMENT_GOALS, OBS_ELEMENT_INDICES
import frames
from tasks import CLOSE_TASKS, TASKS


class RealTimeWrapper(Wrapper):
    """Holds each step to the sim time it covers, so a human-mode window shows the arm at real speed.

    MuJoCo's viewer doesn't pace itself (its timing also ignores frame_skip), so
    without this the arm moves as fast as the policy and the drawing allow. A
    step that already took longer, e.g. one where the model ran, isn't padded.
    """

    def __init__(self, env):
        super().__init__(env)
        self.dt = env.unwrapped.robot_env.dt  # 40 physics steps of 0.002 s
        self.last_step = time.perf_counter()

    def step(self, action):
        result = self.env.step(action)
        time.sleep(max(0.0, self.last_step + self.dt - time.perf_counter()))
        self.last_step = time.perf_counter()
        return result


class HeldSetpointWrapper(Wrapper):
    """Stops the arm from sagging under gravity.

    FrankaKitchen reads the action as a joint velocity and integrates it into a
    position setpoint for the actuators -- but it integrates onto the *measured*
    qpos every step (see _ctrl_velocity_limits in franka_env.py). Gravity droops
    each joint slightly below its setpoint, that droop gets read back in, and the
    next setpoint starts from the drooped position, so the arm ratchets downward:
    ~61 degrees of drift over 200 steps of zero action.

    Holding the setpoint we actually commanded turns that drift into a bounded
    ~1 degree offset, which is what a position servo should do.

    max_lead caps how far the setpoint may run ahead of the arm. Without it the
    setpoint outruns the joints during a fast move and they keep coasting for
    ~20 steps after you release the stick. It must stay above the servo's own
    steady-state droop (~0.02 rad) or the clamp re-anchors to measured qpos and
    the sag comes back. Measured at 0.05: sag 1.4 deg, coast 2.9 deg.
    """

    def __init__(self, env, max_lead=0.05):
        super().__init__(env)
        self.max_lead = max_lead

    def step(self, action):
        result = self.env.step(action)
        env = self.unwrapped
        # data.ctrl is the setpoint the env just commanded, post-clipping.
        qpos = env.data.qpos[:9]
        setpoint = np.clip(env.data.ctrl[:9], qpos - self.max_lead, qpos + self.max_lead)
        env.robot_env._last_robot_qpos = setpoint.copy()
        return result


class VLAObservationWrapper(ObservationWrapper):
    """The free camera at image_size plus joint positions, for a vision-language-action policy.

    Renders at render_size and reduces through frames.resize, the same call load_data
    makes on shards, so a rollout sees exactly what training saw. Object poses are
    left out: the policy reads the scene from the image and the task from the prompt.
    """

    def __init__(self, env, image_size, render_size=frames.RENDER_SIZE):
        super().__init__(env)
        frames.validate(render_size, image_size)  # fail here, not on the first step
        self.image_size = image_size
        # The action is one velocity per joint, so its width is the joint count.
        self.n_joints = env.action_space.shape[0]
        self._set_render_size(env.unwrapped, render_size)
        self.observation_space = spaces.Dict({
            "camera_scene": spaces.Box(0, 255, (image_size, image_size, 3), np.uint8),
            "joint_pos": spaces.Box(-np.inf, np.inf, (self.n_joints,), np.float32),
        })

    def _set_render_size(self, kitchen, render_size):
        """Raise the render resolution above the env's 480 default; must run before anything renders."""
        robot = kitchen.robot_env
        renderer = robot.mujoco_renderer
        assert not renderer._viewers, "render size must be set before the first render"
        # The offscreen viewer sizes itself from renderer.width/height, not offwidth, so both are raised.
        robot.width = robot.height = render_size
        renderer.width = renderer.height = render_size
        kitchen.model.vis.global_.offwidth = render_size
        kitchen.model.vis.global_.offheight = render_size
        self._renderer = renderer

    def observation(self, observation):
        # Offscreen through a second viewer, so this works while render_mode='human' drives a window.
        self._renderer.camera_id = -1  # free camera, the view the human window shows
        frame = self._renderer.render("rgb_array")
        return {
            "camera_scene": frames.resize(frame, self.image_size),
            "joint_pos": observation["observation"][:self.n_joints].astype(np.float32),
        }


class ArmHomeWrapper(Wrapper):
    """Drives the arm back to its start pose without touching the objects.

    The kitchen's reset_model is deterministic -- every episode starts from one
    literal qpos -- so a policy trained on demos only ever sees one arm pose at
    step 0. Rather than teach it to start from anywhere, we make the world keep
    its promise: after a task is done, walk the arm home and leave the scene as
    the task left it. The training distribution then matches inference.

    This is a scripted move, not a learned one. It steps the env with a
    proportional joint-velocity command, so the motion is physical: the sim
    resolves contacts, the camera sees it, and the same routine would port to a
    real arm. Teleporting qpos would be faster and would also let the gripper
    pass through a door it had just opened.

    Homing runs in three phases. The gripper opens in place, so it lets go of a
    handle before anything moves. The hand then backs straight toward the robot
    base, clear of the door it just opened. Only then do the joints drive home.
    Going straight to the joint move swings the arm through a hinge door's arc
    and pushes it half shut (0.3 -> 0.5 from goal, outside the 0.3 success band).

    Homing steps are real env steps, so they count against max_episode_steps.
    Chained sessions need that raised.
    """

    def __init__(self, env, kp=5.0, tolerance=0.02, max_steps=500, release_steps=15,
                 retreat_distance=0.55, retreat_speed=0.5, retreat_max_steps=250, damping=0.05):
        super().__init__(env)
        # Snapshot now: reset_model reads this attribute every reset, so anything
        # that perturbs start poses later would otherwise move our target too.
        self.home = np.array(env.unwrapped.robot_env.init_qpos[:9], dtype=float)
        self.kp = kp
        self.tolerance = tolerance
        self.max_steps = max_steps
        self.release_steps = release_steps
        self.retreat_distance = retreat_distance  # metres the hand backs toward the base
        self.retreat_speed = retreat_speed  # m/s
        self.retreat_max_steps = retreat_max_steps
        self.damping = damping  # damped least squares, keeps the retreat sane near singularities

    def _hand_position(self):
        return self.unwrapped.data.site("end_effector").xpos.copy()

    def _retreat_action(self, direction):
        """Joint velocities that move the hand along direction, fingers held open."""
        env = self.unwrapped
        jacp = np.zeros((3, env.model.nv))
        mujoco.mj_jacSite(env.model, env.data, jacp, None, env.model.site("end_effector").id)
        jac = jacp[:, :7]
        velocity = self.retreat_speed * direction
        qdot = jac.T @ np.linalg.solve(jac @ jac.T + self.damping ** 2 * np.eye(3), velocity)
        action = np.ones(9)
        action[:7] = np.clip(qdot / env.robot_env.act_rng[:7], -1.0, 1.0)
        return action

    def home_actions(self):
        """Yields actions toward home until the arm is there; the caller steps each one."""
        env = self.unwrapped
        for _ in range(self.release_steps):
            yield np.r_[np.zeros(7), np.ones(2)]  # fingers open at full speed, arm holds
        start = self._hand_position()
        direction = np.r_[(env.data.body("panda0_link0").xpos - start)[:2], 0.0]
        direction /= np.linalg.norm(direction)
        for _ in range(self.retreat_max_steps):
            if np.dot(self._hand_position() - start, direction) >= self.retreat_distance:
                break
            yield self._retreat_action(direction)
        while np.abs(self.home - env.data.qpos[:9]).max() >= self.tolerance:
            # act_rng converts a joint velocity in rad/s into the [-1, 1] action.
            yield np.clip(self.kp * (self.home - env.data.qpos[:9]) / env.robot_env.act_rng[:9], -1.0, 1.0)


class CloseTaskWrapper(Wrapper):
    """Starts the scene with one cabinet open and reports "close <task>" once it is shut again.

    The door is set straight to the open task's goal and the arm is left at home,
    so the scene looks like one a chained demo leaves behind (demos end the slide
    at 0.34 and the hinge at 1.55, against goals of 0.37 and 1.45). The env has no
    close goals, so completion is checked here: within band_fraction of the
    closed-to-open span from closed. The env's own 0.3 band would count the slide,
    whose whole span is 0.37, as closed after barely moving.
    """

    def __init__(self, env, open_task, band_fraction):
        super().__init__(env)
        self.task = f"close {open_task}"
        self.indices = OBS_ELEMENT_INDICES[open_task]
        self.closed = np.array(env.unwrapped.robot_env.init_qpos[self.indices], dtype=float)
        self.open = OBS_ELEMENT_GOALS[open_task]
        self.band = band_fraction * np.linalg.norm(self.open - self.closed)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        kitchen = self.unwrapped
        kitchen.data.qpos[self.indices] = self.open
        kitchen.data.qvel[self.indices] = 0.0
        mujoco.mj_forward(kitchen.model, kitchen.data)
        obs, _, _, _, info = self.step(np.zeros(self.action_space.shape[0]))  # one held step, so obs shows the open door
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if np.linalg.norm(self.unwrapped.data.qpos[self.indices] - self.closed) < self.band:
            info["step_task_completions"] = info["step_task_completions"] + [self.task]
        return obs, reward, terminated, truncated, info


def make_kitchen_env(tasks, render_mode, max_episode_steps, image_size, **kitchen_args):
    """FrankaKitchen with held setpoints and VLA observations; real-time pacing when a window shows it."""
    env = gym.make("FrankaKitchen-v1", max_episode_steps=max_episode_steps,
                   tasks_to_complete=tasks, render_mode=render_mode, **kitchen_args)
    if render_mode == "human":
        env = RealTimeWrapper(env)
    env = HeldSetpointWrapper(env)
    return VLAObservationWrapper(env, image_size=image_size)


def make_session_env(render_mode, max_episode_steps, image_size):
    """An env for many commands in a row: it never ends on success and tracks every task."""
    env = make_kitchen_env(list(TASKS), render_mode, max_episode_steps, image_size,
                           terminate_on_tasks_completed=False, remove_task_when_completed=False)
    return ArmHomeWrapper(env)


def make_test_env(task, render_mode, max_episode_steps, image_size, close_band_fraction=0.2):
    """A session env for one test rollout; a CLOSE_TASKS task starts with its cabinet open.

    close_band_fraction: a close task is done within this fraction of the door's span from shut."""
    env = make_session_env(render_mode, max_episode_steps, image_size)
    if task in CLOSE_TASKS:
        env = CloseTaskWrapper(env, CLOSE_TASKS[task], close_band_fraction)
    return env
