import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F

from garage.torch.distributions import TanhNormal
from iod.dads import DADS
from iod import sac_utils


class DADSPoEPolicyModule(nn.Module):
    """State expert, state-option gate Gaussian PoE actor module for DADS."""

    def __init__(
            self,
            *,
            input_dim,
            output_dim,
            dim_option,
            num_heads=4,
            hidden_sizes=(256, 256),
            hidden_nonlinearity=F.relu,
            temperature=1.0,
            init_std=1.0,
            min_std=1e-4,
            max_std=math.exp(2.0),
            final_min_std=1e-4,
            final_max_std=math.exp(2.0),
    ):
        super().__init__()

        if num_heads < 2:
            raise ValueError('DADSPoEPolicyModule requires at least two heads.')
        if dim_option <= 0 or dim_option >= input_dim:
            raise ValueError('dim_option must split a concatenated obs-option input.')
        if temperature <= 0:
            raise ValueError('temperature must be positive.')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dim_option = dim_option
        self.obs_dim = input_dim - dim_option
        self.num_heads = num_heads
        self.hidden_nonlinearity = hidden_nonlinearity
        self.temperature = temperature
        self.min_std = min_std
        self.max_std = max_std
        self.final_min_std = final_min_std
        self.final_max_std = final_max_std

        self.expert_trunk = self._build_mlp(self.obs_dim, hidden_sizes)
        expert_last_dim = hidden_sizes[-1] if hidden_sizes else self.obs_dim
        self.expert_mean = nn.Linear(expert_last_dim, num_heads * output_dim)
        self.expert_log_std = nn.Linear(expert_last_dim, num_heads * output_dim)

        self.gate_trunk = self._build_mlp(input_dim, hidden_sizes)
        gate_last_dim = hidden_sizes[-1] if hidden_sizes else input_dim
        self.gate_logits = nn.Linear(gate_last_dim, num_heads)

        self._init_linear(self.expert_mean)
        self._init_linear(self.expert_log_std)
        self._init_linear(self.gate_logits)
        nn.init.constant_(self.expert_log_std.bias, math.log(init_std))

    @staticmethod
    def _init_linear(layer):
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)

    def _build_mlp(self, input_dim, hidden_sizes):
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layer = nn.Linear(last_dim, hidden_dim)
            self._init_linear(layer)
            layers.append(layer)
            last_dim = hidden_dim
        return nn.ModuleList(layers)

    def _apply_trunk(self, x, trunk):
        for layer in trunk:
            x = layer(x)
            if self.hidden_nonlinearity is not None:
                x = self.hidden_nonlinearity(x)
        return x

    def _split_obs_option(self, inputs):
        obs = inputs[..., :self.obs_dim]
        option = inputs[..., self.obs_dim:]
        return obs, option

    def _expert_stats(self, obs):
        expert_h = self._apply_trunk(obs, self.expert_trunk)
        means = self.expert_mean(expert_h).view(-1, self.num_heads, self.output_dim)
        log_stds = self.expert_log_std(expert_h).view(-1, self.num_heads, self.output_dim)
        log_stds = log_stds.clamp(min=math.log(self.min_std), max=math.log(self.max_std))
        variances = torch.exp(2.0 * log_stds)
        return means, log_stds, variances

    def _combine_experts(self, means, log_stds, variances, weights, gate_logits=None):
        weights = weights.clamp(min=0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        precisions = variances.reciprocal()
        weighted_precisions = weights.unsqueeze(-1) * precisions
        final_precision = weighted_precisions.sum(dim=1).clamp(
            min=1.0 / (self.final_max_std ** 2),
            max=1.0 / (self.final_min_std ** 2),
        )
        final_variance = final_precision.reciprocal()
        final_mean = final_variance * (weighted_precisions * means).sum(dim=1)
        final_std = final_variance.sqrt().clamp(
            min=self.final_min_std,
            max=self.final_max_std,
        )

        return {
            'mean': final_mean,
            'std': final_std,
            'weights': weights,
            'head_means': means,
            'head_log_stds': log_stds,
            'head_variances': variances,
            'gate_logits': gate_logits,
        }

    def _poe_stats(self, inputs):
        obs, _ = self._split_obs_option(inputs)
        means, log_stds, variances = self._expert_stats(obs)

        gate_h = self._apply_trunk(inputs, self.gate_trunk)
        logits = self.gate_logits(gate_h)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        return self._combine_experts(means, log_stds, variances, weights, gate_logits=logits)

    def _poe_stats_with_weights(self, obs, weights):
        means, log_stds, variances = self._expert_stats(obs)
        return self._combine_experts(means, log_stds, variances, weights)

    def forward(self, inputs):
        stats = self._poe_stats(inputs)
        return TanhNormal(stats['mean'], stats['std'])

    def forward_mode(self, inputs):
        stats = self._poe_stats(inputs)
        return torch.tanh(stats['mean'])

    def forward_mode_with_weights(self, obs, weights):
        stats = self._poe_stats_with_weights(obs, weights)
        return torch.tanh(stats['mean'])

    def forward_with_weights(self, obs, weights):
        stats = self._poe_stats_with_weights(obs, weights)
        return TanhNormal(stats['mean'], stats['std'])

    def poe_regularization_terms(self, inputs, head_kl_margin):
        stats = self._poe_stats(inputs)
        means = stats['head_means']
        variances = stats['head_variances']
        weights = stats['weights']

        pair_losses = []
        pair_distances = []
        for i, j in itertools.combinations(range(self.num_heads), 2):
            sym_kl = self._symmetric_diag_gaussian_kl(
                means[:, i],
                variances[:, i],
                means[:, j],
                variances[:, j],
            )
            pair_distances.append(sym_kl)
            pair_losses.append(torch.relu(head_kl_margin - sym_kl))

        if pair_losses:
            head_loss = torch.stack(pair_losses, dim=0).mean()
            head_sym_kl = torch.stack(pair_distances, dim=0).mean()
        else:
            head_loss = inputs.new_tensor(0.0)
            head_sym_kl = inputs.new_tensor(0.0)

        mean_resp = weights.mean(dim=0)
        uniform_resp = torch.full_like(mean_resp, 1.0 / self.num_heads)
        resp_loss = torch.square(mean_resp - uniform_resp).sum()

        normal_entropy = (
            0.5 * (1.0 + math.log(2.0 * math.pi)) + stats['std'].log()
        ).sum(dim=-1)
        entropy_loss = -normal_entropy.mean()

        return {
            'head_loss': head_loss,
            'head_sym_kl': head_sym_kl,
            'resp_loss': resp_loss,
            'entropy_loss': entropy_loss,
            'entropy': normal_entropy.mean(),
            'responsibility': mean_resp,
            'gate_entropy': -(weights * (weights + 1e-8).log()).sum(dim=-1).mean(),
        }

    @staticmethod
    def _symmetric_diag_gaussian_kl(mean_i, var_i, mean_j, var_j):
        delta_sq = torch.square(mean_i - mean_j)
        kl_ij = 0.5 * (
            (var_j.log() - var_i.log())
            + (var_i + delta_sq) / var_j
            - 1.0
        ).sum(dim=-1)
        kl_ji = 0.5 * (
            (var_i.log() - var_j.log())
            + (var_j + delta_sq) / var_i
            - 1.0
        ).sum(dim=-1)
        return 0.5 * (kl_ij + kl_ji)


class DADSPoE(DADS):
    """DADS with a Product-of-Experts primitive-composition actor."""

    def __init__(
            self,
            *,
            poe_head_diversity_coef=0.0,
            poe_responsibility_coef=0.0,
            poe_entropy_coef=0.0,
            poe_head_kl_margin=0.5,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.poe_head_diversity_coef = poe_head_diversity_coef
        self.poe_responsibility_coef = poe_responsibility_coef
        self.poe_entropy_coef = poe_entropy_coef
        self.poe_head_kl_margin = poe_head_kl_margin

    def _update_loss_op(self, tensors, v):
        processed_cat_obs = self._get_concat_obs(
            self.option_policy.process_observations(v['obs']),
            v['options'],
        )
        sac_utils.update_loss_sacp(
            self,
            tensors,
            v,
            obs=processed_cat_obs,
            policy=self.option_policy,
        )

        module = getattr(self.option_policy, '_module', None)
        if not hasattr(module, 'poe_regularization_terms'):
            raise TypeError(
                'DADSPoE expects option_policy._module to implement '
                'poe_regularization_terms().'
            )

        regs = module.poe_regularization_terms(
            processed_cat_obs,
            head_kl_margin=self.poe_head_kl_margin,
        )
        reg_loss = (
            self.poe_head_diversity_coef * regs['head_loss']
            + self.poe_responsibility_coef * regs['resp_loss']
            + self.poe_entropy_coef * regs['entropy_loss']
        )
        tensors['LossSacp'] = tensors['LossSacp'] + reg_loss

        tensors.update({
            'PoEHeadLoss': regs['head_loss'],
            'PoEHeadSymKl': regs['head_sym_kl'],
            'PoERespLoss': regs['resp_loss'],
            'PoEEntropyLoss': regs['entropy_loss'],
            'PoEEntropy': regs['entropy'],
            'PoEGateEntropy': regs['gate_entropy'],
            'PoEResponsibility': regs['responsibility'],
            'PoERegLoss': reg_loss,
        })
