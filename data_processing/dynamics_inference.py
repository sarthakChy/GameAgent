"""Inference and rollout utilities for latent dynamics model."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from data_processing.latent_dynamics_model import LatentDynamicsModel


class LatentDynamicsRollout:
    """Utility for rolling out the latent dynamics model."""

    def __init__(
        self,
        model: LatentDynamicsModel,
        tokenizer,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def rollout(
        self,
        z_init: torch.Tensor,
        action_sequence: list[str],
        max_action_tokens: int = 128,
        deterministic: bool = False,
    ) -> dict:
        """
        Roll out the dynamics model for a sequence of actions.

        Args:
            z_init: [D] initial latent state
            action_sequence: list of action strings
            max_action_tokens: max tokens to tokenize each action
            deterministic: if True, use mean; if False, sample from distribution

        Returns:
            dict with keys:
                - z_trajectory: [T, D] sequence of latent states
                - mu_trajectory: [T, D] sequence of predicted means
                - logvar_trajectory: [T, D] sequence of predicted log-vars
        """
        self.model.eval()
        device = self.device

        z_init = z_init.to(device)
        z_trajectory = [z_init.detach().cpu().unsqueeze(0)]  # [1, D]
        mu_trajectory = []
        logvar_trajectory = []

        z_current = z_init.unsqueeze(0)  # [1, D]

        with torch.no_grad():
            for action_str in action_sequence:
                # Tokenize action
                action_ids = self.tokenizer.encode(
                    action_str,
                    max_length=max_action_tokens,
                    pad_to_max_length=True,
                )
                action_ids = action_ids.unsqueeze(0).to(device)  # [1, T]
                action_length = torch.tensor(
                    [int(action_ids.ne(self.tokenizer.pad_id).sum().item())],
                    device=device,
                )

                # Forward through dynamics model
                mu_pred, logvar_pred = self.model(z_current, action_ids, action_length)

                # Store predictions
                mu_trajectory.append(mu_pred.squeeze(0).detach().cpu())
                logvar_trajectory.append(logvar_pred.squeeze(0).detach().cpu())

                # Sample or use deterministic next state
                if deterministic:
                    z_next = mu_pred
                else:
                    std = torch.exp(0.5 * logvar_pred)
                    eps = torch.randn_like(std)
                    z_next = mu_pred + eps * std

                z_trajectory.append(z_next.detach().cpu())
                z_current = z_next.to(device)

        return {
            "z_trajectory": torch.cat(z_trajectory, dim=0),  # [T+1, D]
            "mu_trajectory": torch.stack(mu_trajectory, dim=0),  # [T, D]
            "logvar_trajectory": torch.stack(logvar_trajectory, dim=0),  # [T, D]
        }

    def compute_trajectory_likelihood(
        self,
        z_init: torch.Tensor,
        action_sequence: list[str],
        z_target_trajectory: torch.Tensor,
        max_action_tokens: int = 128,
    ) -> dict:
        """
        Compute likelihood of observed trajectory under learned dynamics.

        Args:
            z_init: [D] initial state
            action_sequence: list of actions
            z_target_trajectory: [T, D] observed trajectory
            max_action_tokens: max tokens per action

        Returns:
            dict with NLL and other stats
        """
        rollout_result = self.rollout(
            z_init, action_sequence, max_action_tokens, deterministic=False
        )

        mu_traj = rollout_result["mu_trajectory"]  # [T, D]
        logvar_traj = rollout_result["logvar_trajectory"]  # [T, D]
        z_target_trajectory = z_target_trajectory.to(mu_traj.device)

        # Assume target trajectory is ground truth (Dirac)
        # Compute NLL under learned Gaussian
        # NLL = -log p(z_target | mu, logvar)
        nll = 0.5 * (
            torch.log(2 * torch.tensor(3.14159265))
            + logvar_traj
            + (z_target_trajectory - mu_traj) ** 2 / (torch.exp(logvar_traj) + 1e-8)
        )
        nll = nll.sum(dim=-1).mean()  # Average over time and latent dims

        # MSE and cosine similarity for reference
        mse = F.mse_loss(mu_traj, z_target_trajectory)
        cosine_sim = F.cosine_similarity(mu_traj, z_target_trajectory).mean()

        return {
            "nll": float(nll.item()),
            "mse": float(mse.item()),
            "cosine_similarity": float(cosine_sim.item()),
        }

    @torch.no_grad()
    def plan_trajectory(
        self,
        z_init: torch.Tensor,
        z_target: torch.Tensor,
        num_actions: int,
        action_candidates: list[str],
        num_samples: int = 10,
        max_action_tokens: int = 128,
    ) -> dict:
        """
        Simple planning: try different action sequences and pick best.

        Args:
            z_init: [D] current state
            z_target: [D] desired target state
            num_actions: length of action sequence to plan
            action_candidates: list of possible action strings
            num_samples: number of random sequences to try
            max_action_tokens: max tokens per action

        Returns:
            dict with best action sequence and its score
        """
        best_score = float("inf")
        best_actions = None

        for _ in range(num_samples):
            # Random action sequence
            actions = [
                action_candidates[torch.randint(0, len(action_candidates), (1,)).item()]
                for _ in range(num_actions)
            ]

            # Rollout
            rollout_result = self.rollout(z_init, actions, max_action_tokens, deterministic=True)
            z_final = rollout_result["z_trajectory"][-1]  # [D]
            z_target_local = z_target.to(z_final.device)

            # Score: distance to target
            score = float(torch.norm(z_final - z_target_local).item())

            if score < best_score:
                best_score = score
                best_actions = actions

        return {
            "best_actions": best_actions,
            "final_distance_to_target": best_score,
        }
