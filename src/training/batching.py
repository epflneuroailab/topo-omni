"""Modality-homogeneous batching, so image steps can be much larger than video steps.

WHY THIS EXISTS. The spatial loss is a corrcoef over `activations = unified_sheet.reshape(T, -1)`,
and T -- the sample count of that correlation -- is set by the forward's token pooling,
NOT by the batch size. Measured on this checkpoint:

    1 video (20 s, audio)   -> T = 57      (audio-chunk 5 pools ~5 audio tokens per sample)
    1 image                 -> T =  1      (no audio, so ALL visual tokens pool into one)
    4 images                -> T =  4

So an image contributes exactly ONE sample. That is not a defect to route around: it is
the sample axis the orientation eval uses, which correlates units across 638 grating
STIMULI rather than across time-chunks of one clip. To get a usable T from images you
raise the number of images per step, and T comes out equal to the image batch size.

Images are cheap enough to make that practical -- 93 sequence tokens against a video's
3605, and 256 visual patches against 13,156 -- so ~224 images cost about the same as the
current 4-video batch and land on the same T the video path gets at audio-chunk 5.

TWO CONSTRAINTS force a custom sampler rather than a larger per_device_train_batch_size:

  1. Images and videos MUST NOT share a batch. `audio_time_steps_all` in the thinker
     forward is indexed by batch position but only holds entries for items carrying
     audio, and the thinker branch calls .item() on an audio-start-token search that is
     empty for an image. A mixed batch dies with
     "RuntimeError: a Tensor with 0 elements cannot be converted to Scalar".
  2. Every rank must pick the SAME modality at the same optimizer step. Ranks differ in
     graph structure between an image step and a video step, so a disagreement is a DDP
     hang rather than an error.

HOW (2) IS SATISFIED. Accelerate sharding (split_batches=False, which is the Trainer
default) hands CONSECUTIVE underlying batches to consecutive processes -- with 2
processes and a sampler yielding [[0,1,2,3], [4,5,6,7]], process 0 gets [0,1,2,3] and
process 1 gets [4,5,6,7]. So this sampler emits `num_replicas` consecutive batches of one
modality before it switches, and sharding then hands one of them to each rank. It follows
that the sampler must NOT be rank-aware itself; accelerate does the per-rank selection.

The batch sizes differ between image and video steps, which BatchSamplerShard rejects
unless it is built with even_batches=False. Set that via
`SFTConfig(accelerator_config={"even_batches": False})` -- train.py does.
"""
from typing import Optional, Sequence

import numpy as np
from torch.utils.data import Sampler


class ModalityBatchSampler(Sampler):
    """Yields index lists, each drawn from one modality, in runs of `num_replicas`.

    `video_batch_size` / `image_batch_size` are PER RANK; a step therefore consumes
    `batch_size * num_replicas` items globally.

    Args:
        video_indices, image_indices : dataset positions of each modality
        video_batch_size             : per-rank items on a video step
        image_batch_size             : per-rank items on an image step
        seed                         : shuffling seed (shared across ranks by construction,
                                       since every rank builds the identical schedule)
        num_replicas                 : DDP world size
        image_step_ratio             : fraction of OPTIMIZER STEPS that are image steps.
                                       None = whatever the two pool sizes naturally give.
    """

    def __init__(
        self,
        video_indices: Sequence[int],
        image_indices: Sequence[int],
        video_batch_size: int,
        image_batch_size: int,
        seed: int = 42,
        num_replicas: int = 1,
        image_step_ratio: Optional[float] = None,
    ):
        self.video_indices = np.asarray(list(video_indices), dtype=np.int64)
        self.image_indices = np.asarray(list(image_indices), dtype=np.int64)
        self.video_bs = int(video_batch_size)
        self.image_bs = int(image_batch_size)
        self.seed = int(seed)
        self.num_replicas = max(1, int(num_replicas))
        self.epoch = 0
        # Epoch bookkeeping has TWO possible drivers and they must not both apply, or a
        # caller that does use set_epoch gets its epoch double-counted:
        #   - Trainer / accelerate call set_epoch  -> that value is authoritative
        #   - nothing calls it                     -> fall back to counting __iter__,
        #     which is safe because the batch sampler is iterated once per epoch in the
        #     main process and is never handed to dataloader workers.
        self._iter_count = 0
        self._epoch_is_external = False

        # Whole global steps only. A partial step would leave ranks with different batch
        # sizes, and on an image step T (= batch size) would silently change step to step.
        self.n_video_steps = len(self.video_indices) // (self.video_bs * self.num_replicas)
        self.n_image_steps = len(self.image_indices) // (self.image_bs * self.num_replicas)

        if image_step_ratio is not None and self.n_image_steps and self.n_video_steps:
            r = float(image_step_ratio)
            if not 0.0 < r < 1.0:
                raise ValueError(f"image_step_ratio must be in (0, 1), got {r}")
            # Hit the requested mix by capping the over-represented side, not by
            # repeating data inside an epoch.
            by_video = int(round(self.n_video_steps * r / (1.0 - r)))
            if by_video <= self.n_image_steps:
                self.n_image_steps = max(1, by_video)
            else:
                self.n_video_steps = max(1, int(round(self.n_image_steps * (1.0 - r) / r)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._epoch_is_external = True

    def __len__(self) -> int:
        """Batches, not steps: each step emits one batch per rank."""
        return (self.n_video_steps + self.n_image_steps) * self.num_replicas

    def __iter__(self):
        if self._epoch_is_external:
            epoch = self.epoch
        else:
            epoch = self._iter_count
            self._iter_count += 1
        rng = np.random.default_rng(self.seed + 100_003 * epoch)

        vid = rng.permutation(self.video_indices)
        img = rng.permutation(self.image_indices)

        schedule = np.array([0] * self.n_video_steps + [1] * self.n_image_steps)
        rng.shuffle(schedule)

        v_cursor = i_cursor = 0
        for is_image in schedule:
            if is_image:
                span = self.image_bs * self.num_replicas
                chunk, i_cursor, bs = img[i_cursor:i_cursor + span], i_cursor + span, self.image_bs
            else:
                span = self.video_bs * self.num_replicas
                chunk, v_cursor, bs = vid[v_cursor:v_cursor + span], v_cursor + span, self.video_bs
            # One batch per rank, consecutively -- see "HOW (2) IS SATISFIED" above.
            for r in range(self.num_replicas):
                yield chunk[r * bs:(r + 1) * bs].tolist()

    def describe(self) -> str:
        n = self.n_video_steps + self.n_image_steps
        return (f"ModalityBatchSampler: {n} steps/epoch = "
                f"{self.n_video_steps} video ({self.video_bs}/rank) + "
                f"{self.n_image_steps} image ({self.image_bs}/rank), "
                f"world {self.num_replicas}. Spatial-loss samples per rank: "
                f"T ~ {57 * self.video_bs} on a video step (57 per clip at audio-chunk 5), "
                f"T = {self.image_bs} on an image step (1 per image).")


def split_by_modality(dataset, column: str = "modality"):
    """(video_indices, image_indices) from the dataset's modality column.

    Reads the arrow column directly rather than iterating rows: the tokenised dataset
    holds pixel tensors, and __getitem__ over 100k+ rows to read one string field is
    minutes of needless decoding.
    """
    if column not in dataset.column_names:
        # No modality column = the old video-only manifest.
        return list(range(len(dataset))), []
    mods = dataset.with_format(None)[column]
    vid = [i for i, m in enumerate(mods) if m != "image"]
    img = [i for i, m in enumerate(mods) if m == "image"]
    return vid, img
