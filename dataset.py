import numpy as np
import os
import time
from concurrent.futures import ThreadPoolExecutor
import glob
import frames

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
    768 KiB at 512, 588 KiB at 448, 147 KiB at 224, 48 KiB at 128.
    """

    def __init__(self, max_size, image_size, n_actions, n_joints):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.camera_scene_memory = np.zeros(
            (self.mem_size, image_size, image_size, 3), dtype=np.uint8)
        self.joint_pos_memory = np.zeros((self.mem_size, n_joints), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        self.task_description_memory = np.zeros(self.mem_size, dtype=object)
        # Row index one past the last step of each step's episode; action chunks must not cross it.
        self.episode_end_memory = np.zeros(self.mem_size, dtype=np.int64)
        # False for done and fumble frames in version 2 shards, which stay out of the action loss.
        self.action_ok_memory = np.ones(self.mem_size, dtype=bool)

    def load_data(self, path):
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

        if self.mem_ctr + total > self.mem_size:
            raise ValueError(
                f"{path} holds {total} steps across {len(files)} shards, but this "
                f"Dataset was built for {self.mem_size} and has {self.mem_ctr} already. Raise max_size to at least "
                f"{self.mem_ctr + total} ({(self.mem_ctr + total) * self.camera_scene_memory[0].nbytes / 1e9:.1f} GB "
                f"of frames) or point at a smaller directory.")

        index = self.mem_ctr
        # pool.map keeps input order, which is what lines results up with the counts.
        with ThreadPoolExecutor(WORKERS) as pool:
            loaded = pool.map(self._read, files)

            for steps, (camera, data) in zip(steps_per_file, loaded):
                end = index + steps
                self.camera_scene_memory[index:end] = camera
                self.joint_pos_memory[index:end] = data['joint_pos']
                self.action_memory[index:end] = data['action']
                self.terminal_memory[index:end] = data['done']
                self.task_description_memory[index:end] = str(data['task_description'])
                self.episode_end_memory[index:end] = end  # each shard is one episode
                done = data['done'].astype(bool)
                if 'schema_version' in data:
                    self.action_ok_memory[index:end] = ~(done | data['exclude_action'].astype(bool))
                index = end

        self.mem_ctr += total

        elapsed = time.time() - start

        print(f"Loaded {total} steps from {path} in {elapsed:.1f}s, {self.mem_ctr} in all")

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

    def sample_batch(self, batch_size, chunk_size=1):
        """Random usable steps, each with the chunk_size actions starting there. action_mask is False where a chunk runs past its episode or into frames kept out of the action loss."""
        batch = np.random.choice(np.flatnonzero(self.action_ok_memory[:self.mem_ctr]), batch_size)
        episode_end = self.episode_end_memory[batch][:, None]
        steps = batch[:, None] + np.arange(chunk_size)  # (batch, chunk) rows of the actions to predict
        action_mask = steps < episode_end
        steps = np.minimum(steps, episode_end - 1)  # masked steps still need a valid row to read
        action_mask &= self.action_ok_memory[steps]

        state = {
            # NHWC uint8, which is what the VLM processor takes as-is.
            "camera_scene": self.camera_scene_memory[batch],
            "joint_pos": self.joint_pos_memory[batch],
            "row": batch,
        }

        return (state,
                self.action_memory[steps],  # (batch, chunk, actions)
                action_mask,
                # Task description text, not the id: the VLM reads the instruction.
                list(self.task_description_memory[batch]))
