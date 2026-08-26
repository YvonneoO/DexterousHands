import math

import torch

from bidexhands.tasks.shadow_hand_bottle_cap import ShadowHandBottleCap


class ShadowHandBottleCapV2(ShadowHandBottleCap):
    """Bottle-cap task with physically meaningful relative-rotation success.

    The legacy task rewards Euclidean separation between the lid and bottle.  This
    subclass keeps the original simulator, observations, and controller, but uses
    the articulated object's revolute and prismatic DOFs directly for reward and
    success.  Variant-specific parameters live in separate YAML files.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        env_cfg = self.cfg["env"]
        self.v2_variant = env_cfg.get("v2Variant", "rotation_basic")
        self.v2_direction = float(env_cfg.get("v2Direction", 1.0))
        self.v2_rotation_scale = float(env_cfg.get("v2RotationScale", 80.0))
        self.v2_angle_scale = float(env_cfg.get("v2AngleScale", 0.25))
        self.v2_contact_scale = float(env_cfg.get("v2ContactScale", 1.0))
        self.v2_force_contact_scale = float(env_cfg.get("v2ForceContactScale", 0.0))
        self.v2_pull_penalty_scale = float(env_cfg.get("v2PullPenaltyScale", 15.0))
        self.v2_action_penalty_scale = float(env_cfg.get("v2ActionPenaltyScale", 0.0002))
        self.v2_stability_scale = float(env_cfg.get("v2StabilityScale", 0.02))
        self.v2_success_rotation = float(env_cfg.get("v2SuccessRotation", 2.0 * math.pi))
        self.v2_contact_threshold = float(env_cfg.get("v2ContactThreshold", 0.12))
        self.v2_max_early_pull = float(env_cfg.get("v2MaxEarlyPull", 0.006))
        self.v2_success_hold_steps = int(env_cfg.get("v2SuccessHoldSteps", 15))

        self.prev_cap_angle = self.object_dof_pos[:, 0].clone()
        self.cap_angle_origin = self.object_dof_pos[:, 0].clone()
        self.cap_success_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        print("BottleCapV2 variant:", self.v2_variant)
        print("BottleCapV2 direction / success radians:", self.v2_direction, self.v2_success_rotation)

    def reset(self, env_ids, goal_env_ids):
        super().reset(env_ids, goal_env_ids)
        if hasattr(self, "prev_cap_angle"):
            self.prev_cap_angle[env_ids] = self.object_dof_pos[env_ids, 0]
            self.cap_angle_origin[env_ids] = self.object_dof_pos[env_ids, 0]
            self.cap_success_hold[env_ids] = 0

    def compute_reward(self, actions):
        cap_angle = self.object_dof_pos[:, 0]
        cap_slide = self.object_dof_pos[:, 1]
        angle_delta = cap_angle - self.prev_cap_angle
        angle_from_start = cap_angle - self.cap_angle_origin

        if self.v2_direction == 0.0:
            signed_delta = torch.abs(angle_from_start) - torch.abs(self.prev_cap_angle - self.cap_angle_origin)
            rotation_progress = torch.abs(angle_from_start)
        else:
            signed_delta = self.v2_direction * angle_delta
            rotation_progress = self.v2_direction * angle_from_start

        finger_dist_mean = (
            torch.norm(self.bottle_cap_pos - self.right_hand_ff_pos, p=2, dim=-1)
            + torch.norm(self.bottle_cap_pos - self.right_hand_mf_pos, p=2, dim=-1)
            + torch.norm(self.bottle_cap_pos - self.right_hand_rf_pos, p=2, dim=-1)
            + torch.norm(self.bottle_cap_pos - self.right_hand_lf_pos, p=2, dim=-1)
            + torch.norm(self.bottle_cap_pos - self.right_hand_th_pos, p=2, dim=-1)
        ) / 5.0
        body_dist = torch.norm(self.bottle_pos - self.left_hand_pos, p=2, dim=-1)

        cap_proximity = torch.exp(-20.0 * finger_dist_mean)
        body_proximity = torch.exp(-20.0 * body_dist)

        sensor = self.vec_sensor_tensor.view(self.num_envs, self.num_fingertips, 6)
        right_force = torch.norm(sensor[:, :5, :3], p=2, dim=-1).mean(dim=-1)
        left_force = torch.norm(sensor[:, 5:, :3], p=2, dim=-1).mean(dim=-1)
        right_force_contact = torch.tanh(right_force / 10.0)
        left_force_contact = torch.tanh(left_force / 10.0)

        geometric_contact = cap_proximity * body_proximity
        force_contact = right_force_contact * left_force_contact
        contact_gate = torch.where(
            (finger_dist_mean < self.v2_contact_threshold) & (body_dist < 0.16),
            torch.ones_like(rotation_progress),
            torch.zeros_like(rotation_progress),
        )

        rotation_reward = self.v2_rotation_scale * torch.clamp(signed_delta, -0.05, 0.05) * contact_gate
        angle_reward = self.v2_angle_scale * torch.clamp(rotation_progress, 0.0, self.v2_success_rotation)
        contact_reward = self.v2_contact_scale * (cap_proximity + 0.5 * body_proximity)
        contact_reward = contact_reward + self.v2_force_contact_scale * force_contact

        early_pull = torch.relu(torch.abs(cap_slide) - self.v2_max_early_pull)
        early_pull_gate = torch.where(
            rotation_progress < 0.75 * self.v2_success_rotation,
            torch.ones_like(early_pull),
            torch.zeros_like(early_pull),
        )
        pull_penalty = self.v2_pull_penalty_scale * early_pull * early_pull_gate
        action_penalty = self.v2_action_penalty_scale * torch.sum(actions ** 2, dim=-1)
        stability_penalty = self.v2_stability_scale * (
            torch.norm(self.object_linvel, p=2, dim=-1) + torch.norm(self.object_angvel, p=2, dim=-1)
        )

        self.rew_buf[:] = (
            rotation_reward + angle_reward + contact_reward - pull_penalty - action_penalty - stability_penalty
        )

        physically_complete = (
            (rotation_progress >= self.v2_success_rotation)
            & (finger_dist_mean < self.v2_contact_threshold)
            & (body_dist < 0.16)
            & (self.object_pos[:, 2] > 0.45)
        )
        self.cap_success_hold = torch.where(
            physically_complete,
            self.cap_success_hold + 1,
            torch.zeros_like(self.cap_success_hold),
        )
        just_succeeded = self.cap_success_hold >= self.v2_success_hold_steps
        self.successes[:] = torch.where(
            (self.successes == 0) & just_succeeded,
            torch.ones_like(self.successes),
            self.successes,
        )

        resets = self.reset_buf
        resets = torch.where(self.object_pos[:, 2] <= 0.40, torch.ones_like(resets), resets)
        resets = torch.where(body_dist >= 0.30, torch.ones_like(resets), resets)
        resets = torch.where(self.progress_buf >= self.max_episode_length, torch.ones_like(resets), resets)
        self.reset_buf[:] = resets
        self.reset_goal_buf[:] = torch.zeros_like(self.reset_goal_buf)
        self.consecutive_successes[:] = torch.where(
            resets > 0, self.successes * resets, self.consecutive_successes
        )

        self.prev_cap_angle.copy_(cap_angle)
        self.extras["successes"] = self.successes
        self.extras["consecutive_successes"] = self.consecutive_successes
        self.extras["cap_rotation_rad"] = rotation_progress
        self.extras["cap_prismatic_m"] = torch.abs(cap_slide)
        self.extras["cap_geometric_contact"] = geometric_contact
        self.extras["cap_force_contact"] = force_contact


class ShadowHandBottleCapV2A(ShadowHandBottleCapV2):
    pass


class ShadowHandBottleCapV2B(ShadowHandBottleCapV2):
    pass


class ShadowHandBottleCapV2C(ShadowHandBottleCapV2):
    pass
