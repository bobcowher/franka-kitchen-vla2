import numpy as np
import yaml
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
    """A camera view plus proprioception, for a vision-language-action policy.

    Returns {"camera_scene", "joint_pos", "joint_vel"}, matching the
    camera/joint_pos/joint_vel split used by the other buffers in this repo.

    Object poses are deliberately left out. The old wrapper concatenated the
    env's achieved_goal and desired_goal onto a flat joint-state vector, which
    made sense when the policy had no eyes. A VLA reads the scene from the image
    and the task from the language instruction, so both were redundant anyway:
    achieved_goal is a copy of columns already inside `observation`, and
    desired_goal is a per-task constant. They were also the only reason the
    observation width varied by task (61/63/73), which blocked mixing shards.

    camera_scene is the free camera from DEFAULT_CAMERA_CONFIG -- the same view
    the human window shows, so demonstrator and policy see the same thing. The
    model's own 'left_cap' and 'right_cap' cameras point at a countertop and a
    cabinet face; neither shows the arm, so neither is used.

    There is no wrist view. One was built and aimed (see the wrist-camera
    branch), but this environment is a fixed scene with a fixed viewpoint and
    coarse tasks -- doors, knobs, a kettle -- so the conditions that make
    eye-in-hand views pay off in the literature mostly do not apply here. It
    doubled storage for a benefit this setup was unlikely to collect.
    """

    def __init__(self, env, image_size=frames.RENDER_SIZE):
        super().__init__(env)
        self.image_size = image_size
        self._set_render_size(env.unwrapped, image_size)
        self.observation_space = spaces.Dict({
            "camera_scene": spaces.Box(0, 255, (image_size, image_size, 3), np.uint8),
            "joint_pos": spaces.Box(-np.inf, np.inf, (9,), np.float32),
            "joint_vel": spaces.Box(-np.inf, np.inf, (9,), np.float32),
        })

    def _set_render_size(self, kitchen, image_size):
        """Raise the render resolution above the env's 480 default.

        Must run before anything renders, because the viewers read these when
        they are first created.
        """
        robot = kitchen.robot_env
        renderer = robot.mujoco_renderer
        assert not renderer._viewers, (
            "render size must be set before the first render")

        # The offscreen viewer sizes itself from renderer.width/height, not from
        # offwidth, so both have to be raised or we would upscale a 480 render.
        # This also enlarges the teleop window, which is no bad thing.
        robot.width = robot.height = image_size
        renderer.width = renderer.height = image_size
        kitchen.model.vis.global_.offwidth = image_size
        kitchen.model.vis.global_.offheight = image_size

        self._renderer = renderer

    def _grab(self):
        # Renders offscreen through a second viewer, so this works even while
        # render_mode='human' is driving the teleop window.
        self._renderer.camera_id = -1  # free camera, per DEFAULT_CAMERA_CONFIG
        frame = self._renderer.render("rgb_array")
        # Should already be image_size -- _set_render_size asked for it. If it
        # is not, resize the same way the training loader does rather than a
        # second way, so the policy never sees two flavours of the same scene.
        return frames.resize(frame, self.image_size)

    def observation(self, observation):
        robot_obs = observation["observation"]
        return {
            "camera_scene": self._grab(),
            "joint_pos": robot_obs[:9].astype(np.float32),
            "joint_vel": robot_obs[9:18].astype(np.float32),
        }


class ObsReshapeWrapper(ObservationWrapper):
    """Brings a rendered observation down to the size the policy trained on.

    The collector archives at the render size so a shard stays useful to models
    that want more pixels than today's network does. Training reduces those
    shards in load_data; this applies the identical reduction to a live env, so
    a rollout sees what training saw rather than something a few grey levels off
    along every edge.

    Outermost wrapper. It reshapes whatever camera_scene the layer below built,
    so anything that wants full-resolution frames -- recording, debugging --
    goes below it or leaves it off.

    Frames also come out CHW rather than the HWC everything upstream uses,
    because that is what Conv2d takes and what Dataset.sample_batch hands back.
    Both are free views over HWC memory, so nothing is copied. Once a batch
    dimension is added -- obs[None], or a DataLoader collate -- torch reports
    the result channels_last-contiguous, which is the faster layout on GPU;
    channels_last is a 4D format, so an unbatched frame will say False. Storage
    stays HWC: it compresses better and it is what RLDS and every image library
    use.

    Only camera_scene is touched; joint_pos and joint_vel pass through. Named
    for reshaping generally because a crop or a normalisation would belong here
    too -- anything that has to happen identically on both sides of the
    train/eval line.
    """

    def __init__(self, env, image_size):
        super().__init__(env)
        self.image_size = image_size

        # Fail here rather than on the first step of a rollout.
        frames.validate(env.observation_space["camera_scene"].shape[0], image_size)

        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            "camera_scene": spaces.Box(0, 255, (3, image_size, image_size), np.uint8),
        })

    def observation(self, observation):
        reduced = frames.resize(observation["camera_scene"], self.image_size)
        return {
            **observation,
            # CHW, matching Dataset.sample_batch so a policy sees one axis order
            # either side of the train/eval line. A view, not a copy.
            "camera_scene": reduced.transpose(2, 0, 1),
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

    def return_to_home(self):
        """Step until the arm is home. Returns (observation, steps_taken)."""
        env = self.unwrapped
        # act_rng converts a joint velocity in rad/s into the [-1, 1] action.
        act_rng = env.robot_env.act_rng[:9]
        observation = None

        for step in range(self.max_steps):
            error = self.home - env.data.qpos[:9]
            if np.abs(error).max() < self.tolerance:
                return observation, step
            action = np.clip(self.kp * error / act_rng, -1.0, 1.0)
            observation, _, _, _, _ = self.env.step(action)

        print(f"return_to_home gave up after {self.max_steps} steps, "
              f"worst joint still {np.abs(self.home - env.data.qpos[:9]).max():.3f} rad off")
        return observation, self.max_steps


class RandomStartWrapper(Wrapper):
    """Teleports the arm to one of a handful of pre-captured poses at reset.

    reset_model always starts from the same qpos, so a single-start dataset
    never shows a policy how to recover once it drifts off the one
    trajectory it was shown -- exactly what stalled the microwave rollout
    that motivated this. Picking a random, known-good pose from `poses_path`
    gives demos genuinely different starts without risking the collision a
    blind random qpos could land in: every pose in the file was reached by
    driving the real arm there (see scripts/capture_start_poses.py).

    poses_path is YAML, not npy, so a human can read, hand-edit, or
    rebalance it -- a list of {qpos: [9 floats], frequency: weight}. frequency
    is a relative sampling weight, not a probability; weights are normalized
    at load time, so editing one entry's frequency is enough to rebalance,
    without touching the others.

    Objects are left exactly where reset_model put them; only the 9 arm
    joints move, and only at episode start. This is a direct qpos write, not
    a scripted move like ArmHomeWrapper -- nothing has happened yet at this
    point in the episode, so there's no gripper-through-geometry risk from
    teleporting.
    """

    def __init__(self, env, poses_path="start_poses.yaml"):
        super().__init__(env)
        with open(poses_path) as f:
            entries = yaml.safe_load(f)
        assert entries, f"no poses found in {poses_path}"

        self.poses = np.array([entry["qpos"] for entry in entries], dtype=float)
        assert self.poses.ndim == 2 and self.poses.shape[1] == 9, (
            f"expected 9-dim qpos entries in {poses_path}, got shape {self.poses.shape}")

        weights = np.array([entry.get("frequency", 1.0) for entry in entries], dtype=float)
        self.probs = weights / weights.sum()

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)

        kitchen = self.unwrapped
        robot = kitchen.robot_env

        pose = self.poses[np.random.choice(len(self.poses), p=self.probs)]
        qpos = kitchen.data.qpos.copy()
        qvel = kitchen.data.qvel.copy()
        qpos[:9] = pose
        qvel[:9] = 0.0
        robot.set_state(qpos, qvel)

        # env.reset() above already rendered once, showing reset_model's
        # pose -- teleporting afterward leaves that stale frame on screen
        # until the next env.step() happens to render again. Render now so
        # the human window (and anyone driving off of it) shows where the
        # arm actually is, not where it started.
        kitchen.render()

        robot_obs = robot._get_obs()
        return kitchen._get_obs(robot_obs), info
