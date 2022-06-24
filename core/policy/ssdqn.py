from typing import Any, Dict, Union, Optional

import torch
import torch.nn.functional as F
import numpy as np

import cv2
import albumentations as A

from tianshou.policy import BasePolicy, DQNPolicy
from tianshou.data import Batch


class SS_MixIn(BasePolicy):
    def __init__(self, num_directions, regularize_every, aug_probs=[0.7, 0.3, 0], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_directions = num_directions
        self.regularize_every = regularize_every
        self.aug_probs = aug_probs

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        main_logs = super().learn(batch, **kwargs)
        if self._iter % self.regularize_every == 0:
            aux_logs = self._regularization_step(batch, **kwargs)
        else:
            aux_logs = {}
        return dict(**main_logs, **aux_logs)

    def _regularization_step(self, batch: Batch, **kwargs: Any):
        input = batch["obs"]
        obs, actions = input["obs"], input["actions"]

        batch_size = obs.shape[0]
        num_samples = batch_size // 2
        indices = np.random.choice(np.arange(0, batch_size), size=num_samples, replace=False)

        sampled_obs, sampled_actions = obs[indices], actions[indices]
        sampled_rgb = np.transpose(sampled_obs[:, :3], (0, 2, 3, 1))
        sampled_movement = np.transpose(sampled_obs[:, 3:], (0, 2, 3, 1))

        aug_rgb, aug_movement, aug_actions, post_process_udf = self._generate_augmentations(
            sampled_rgb, sampled_movement, sampled_actions
        )
        aug_rgb = np.transpose(aug_rgb, (0, 3, 1, 2))
        aug_movement = np.transpose(aug_movement, (0, 3, 1, 2))
        aug_obs = np.concatenate((aug_rgb, aug_movement), axis=1)

        aug_batch = Batch(
            obs={
                "obs": np.concatenate((sampled_obs, aug_obs), axis=0),
                "actions": np.concatenate((sampled_actions, aug_actions), axis=0),
            },
            info={},
        )

        # optimize
        self.optim.zero_grad()
        q = self(aug_batch).logits
        q_ref, q_aug = q[:num_samples], post_process_udf(q[num_samples:])
        q_ref = q_ref.detach()
        loss = F.mse_loss(q_aug, q_ref)
        loss.backward()
        self.optim.step()
        return {"loss/regularization": loss.item()}

    def _generate_augmentations(self, imgs, masks, actions):
        tfms, input_mapping, post_process = self._get_tfms()

        aug_im, aug_masks = [np.stack(x, axis=0) for x in zip(*[tfms(im, mask) for im, mask in zip(imgs, masks)])]
        aug_actions = np.vectorize(input_mapping.get)(actions)

        return aug_im, aug_masks, aug_actions, post_process

    def _get_tfms(self):
        aug = np.random.choice(["DIR_ROT", "FLIP", "SHIFT"], p=self.aug_probs)
        source = np.arange(0, self.num_directions)

        if aug == "DIR_ROT":
            sign = np.random.choice([-1, 1])
            direction_shift_angle = 360 / self.num_directions
            angle = sign * direction_shift_angle

            _rotate = lambda x: A.rotate(x, angle, interpolation=cv2.INTER_AREA, border_mode=cv2.BORDER_REFLECT_101)
            tfms = lambda im, mask: (_rotate(im), _rotate(mask))
            target = np.roll(source, -sign)

        elif aug == "FLIP":
            sign = np.floor(np.random.uniform(high=4)).astype(int)
            if sign == 0:  # HORIZONTAL FLIP
                op = lambda im: im[:, ::-1]
                split = self.num_directions // 2
                target = np.concatenate((source[: split + 1][::-1], source[split + 1 :][::-1]))  # nasty reverse!
            else:
                op = lambda im: np.rot90(im, k=sign)
                target = np.roll(source, -sign * self.num_directions // 4)
            tfms = lambda im, mask: (op(im), op(mask))

        elif aug == "SHIFT":
            op = A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0,
                rotate_limit=0,
                interpolation=cv2.INTER_AREA,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1,
            )
            tfms = lambda im, mask: map(op(image=im, mask=mask).get, ["image", "mask"])
            target = source

        input_mapping = dict(np.concatenate([[(0, 0), (1, 1)], np.stack([source, target], axis=1) + 2]))
        post_process = lambda x: torch.cat([x[:, :1], x[:, 1:][:, target]], dim=1)

        return tfms, input_mapping, post_process


SS_DQN = type("SS_DQN", (SS_MixIn, DQNPolicy), {})
