import numpy as np
from gymnasium import ObservationWrapper, Wrapper, spaces
import frames


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

    The gripper fingers are homed too, which means anything held gets released.
    That is what a "ready for the next task" pose should do.

    Homing steps are real env steps, so they count against max_episode_steps.
    Chained sessions need that raised.
    """

    def __init__(self, env, kp=5.0, tolerance=0.02, max_steps=150):
        super().__init__(env)
        # Snapshot now: reset_model reads this attribute every reset, so anything
        # that perturbs start poses later would otherwise move our target too.
        self.home = np.array(env.unwrapped.robot_env.init_qpos[:9], dtype=float)
        self.kp = kp
        self.tolerance = tolerance
        self.max_steps = max_steps

    def home_action(self):
        """The next action toward home, or None once the arm is there."""
        env = self.unwrapped
        error = self.home - env.data.qpos[:9]
        if np.abs(error).max() < self.tolerance:
            return None
        # act_rng converts a joint velocity in rad/s into the [-1, 1] action.
        return np.clip(self.kp * error / env.robot_env.act_rng[:9], -1.0, 1.0)
