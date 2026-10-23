import numpy as np
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import glob
import frames
from tasks import task_index

# Inflate and resize both release the GIL, so threads scale here where they
# normally would not. Measured 8x on 32 cores, flat past 16.
WORKERS = min(16, os.cpu_count() or 1)


def _open(filename):
    """np.load, but say which shard is bad rather than raising from zipfile."""
    try:
        return np.load(filename)
    except Exception as e:
        raise ValueError(
            f"{filename} could not be read ({type(e).__name__}). A shard left "
            f"half-written by a killed collector looks like this; delete it.") from e


class Dataset():
    """Demonstrations loaded off disk and sampled in batches for behavior cloning.

    The arenas mirror the observation space in gym_robotics_custom.py, so a
    policy written against the live env works unchanged on the dataset.

    load_data raises rather than growing past max_size, so a directory holding
    more than you planned for is an error instead of a machine that swaps.

    Shards are archived at 896 and reduced to image_size on the way in, which is
    the lever that keeps a large dataset in memory: a step costs 2.30 MiB at 896,
    147 KiB at 224, 48 KiB at 128.
    """

    def __init__(self, max_size, image_size, n_actions, n_joints=9):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.camera_scene_memory = np.zeros(
            (self.mem_size, image_size, image_size, 3), dtype=np.uint8)
        self.joint_pos_memory = np.zeros((self.mem_size, n_joints), dtype=np.float32)
        self.joint_vel_memory = np.zeros((self.mem_size, n_joints), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.task_description_memory = np.zeros(self.mem_size, dtype=object)
        self.task_id_memory = np.zeros(self.mem_size, dtype=np.int64)

    def __len__(self):
        return self.mem_ctr

    def can_sample(self, batch_size):
        return self.mem_ctr >= batch_size

    def load_data(self, path, verbose=False):
        """Load every npz shard under path, recursively."""
        start = time.time()
        print(f"Loading data from {path}...")

        files = sorted(glob.glob(os.path.join(path, '**', '*.npz'), recursive=True))

        if not files:
            print(f"No npz files found under {path}")
            return

        # Counted first, so an oversized directory fails before anything is written.
        steps_per_file = [len(_open(filename)['action']) for filename in files]
        total = sum(steps_per_file)

        if total > self.mem_size:
            raise ValueError(
                f"{path} holds {total} steps across {len(files)} shards, but this "
                f"Dataset was built for {self.mem_size}. Raise max_size to at least "
                f"{total} ({total * self.camera_scene_memory[0].nbytes / 1e9:.1f} GB "
                f"of frames) or point at a smaller directory.")

        index = 0
        # pool.map keeps input order, which is what lines results up with the counts.
        with ThreadPoolExecutor(WORKERS) as pool:
            loaded = pool.map(self._read, files)

            for filename, steps, (camera, data) in zip(
                    files, steps_per_file, loaded):
                if verbose:
                    print(f"  {filename}: {steps} steps")

                end = index + steps
                self.camera_scene_memory[index:end] = camera
                self.joint_pos_memory[index:end] = data['joint_pos']
                self.joint_vel_memory[index:end] = data['joint_vel']
                self.action_memory[index:end] = data['action']
                self.reward_memory[index:end] = data['reward']
                self.terminal_memory[index:end] = data['done']
                description = str(data['task_description'])
                self.task_description_memory[index:end] = description
                self.task_id_memory[index:end] = task_index(description)[0]
                index = end

        self.mem_ctr = total

        elapsed = time.time() - start

        print(f"Loaded {self.mem_ctr} steps from {path} in {elapsed:.1f}s")

    def _read(self, filename):
        """Decompress one shard and reduce its frames, on a worker thread.

        Resizing here rather than in the caller keeps the expensive half
        parallel and frees the full-size frames sooner.
        """
        data = _open(filename)
        try:
            camera = frames.resize(data['camera_scene'],
                                   self.camera_scene_memory.shape[1])
        except ValueError as e:
            # resize only sees an array, so it cannot name the odd shard.
            raise ValueError(f"{filename}: {e}") from None
        return camera, data

    def sample_batch(self, batch_size):
        batch = np.random.choice(self.mem_ctr, batch_size)

        state = {
            # NCHW to match Conv2d. Fancy indexing already made a contiguous HWC
            # copy, so the transpose is a free view over it.
            "camera_scene": self.camera_scene_memory[batch].transpose(0, 3, 1, 2),
            "joint_pos": self.joint_pos_memory[batch],
            "joint_vel": self.joint_vel_memory[batch],
        }

        return (state,
                self.action_memory[batch],
                self.reward_memory[batch],
                self.terminal_memory[batch],
                self.task_id_memory[batch])


class DatasetShard():
    """One collection run's worth of VLA transitions, saved as its own file.

    Overrunning max_size raises rather than wrapping: a shard is a single
    episode, and a ring buffer would corrupt it instead of ageing out old data.

    Frames go in HWC, straight off the renderer. Do not record through
    ObsReshapeWrapper -- it hands out NCHW and the arena would reject the shape.

    Measured over 104 microwave demos: 509 KiB per timestep, 59 steps per demo,
    3.2 GB. Demo length drives the total, not the task count.

    No next_state is stored. Behavior cloning never reads it, and within a shard
    it is an exact duplicate of the following row.
    """

    def __init__(self, max_size, image_size, n_actions, task_name,
                 task_description, n_joints=9):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.task_name = task_name
        self.task_description = task_description
        self.camera_scene_memory = np.zeros(
            (self.mem_size, image_size, image_size, 3), dtype=np.uint8)
        self.joint_pos_memory = np.zeros((self.mem_size, n_joints), dtype=np.float32)
        self.joint_vel_memory = np.zeros((self.mem_size, n_joints), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def __len__(self):
        return self.mem_ctr

    def store_transition(self, state, action, reward, done):
        if self.mem_ctr >= self.mem_size:
            raise ValueError(
                f"shard is full at {self.mem_size} steps. Raise max_size to match "
                f"max_episode_steps, or the episode is running longer than expected.")

        index = self.mem_ctr

        self.camera_scene_memory[index] = state["camera_scene"]
        self.joint_pos_memory[index] = state["joint_pos"]
        self.joint_vel_memory[index] = state["joint_vel"]
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_ctr += 1

    def save_to_csv(self):

        date_formatted = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        directory = f'dataset/{self.task_name}'
        filename = f'{directory}/shard_{date_formatted}.npz'

        os.makedirs(directory, exist_ok=True)

        if os.path.exists(filename):
            raise FileExistsError(
                f"{filename} already exists. Shard names are second-resolution, "
                f"so this shard would overwrite one saved in the same second.")

        # savez_compressed leaves an invalid zip for the ~1s it runs, so write to
        # a temp name and os.replace it into place. A reader then sees the whole
        # shard or no file, which makes it safe to load a directory while a
        # collection run is still appending to it.
        tmp = f'{filename}.tmp'
        with open(tmp, 'wb') as fh:
            np.savez_compressed(fh,
                     task_name=self.task_name,
                     task_description=self.task_description,
                     camera_scene=self.camera_scene_memory[:self.mem_ctr],
                     joint_pos=self.joint_pos_memory[:self.mem_ctr],
                     joint_vel=self.joint_vel_memory[:self.mem_ctr],
                     action=self.action_memory[:self.mem_ctr],
                     reward=self.reward_memory[:self.mem_ctr],
                     done=self.terminal_memory[:self.mem_ctr])
        os.replace(tmp, filename)

        print("-" * 20)
        print(f"Saved {filename} ({self.mem_ctr} steps, "
              f"{os.path.getsize(filename) / 1e6:.1f} MB)")
        print("-" * 20)
