# DADS-Based PoE Meta-Skill Implementation Plan

## Goal

Implement a DADS-based unsupervised skill discovery method where the original fixed skill latent `z` is preserved as the long-horizon exploration intent, but the actor internally composes reusable primitive policy heads through a Product-of-Experts (PoE) structure.

The main hypothesis is:

> Existing USD methods use `z` as an outcome-level skill code. We instead use `z` to condition a state-dependent composition over primitive experts, so that diverse state coverage is produced through reusable internal motor components.

## Starting Point

Use the existing DADS codebase as the base.

Keep:

- DADS skill sampling schedule
- fixed `z` over the skill horizon
- DADS intrinsic reward / dynamics-predictability objective
- replay buffer format, unless actor changes require minor logging additions
- downstream evaluation protocol if already available

Modify:

- the actor architecture
- actor log-prob computation
- optional regularization losses for expert diversity and responsibility balance

## Core Policy Architecture

Replace the standard skill-conditioned Gaussian actor:

```latex
\pi(a \mid s, z)
```

with a multi-head Gaussian PoE actor:

```latex
\pi(a_t \mid s_t, z)
\propto
\prod_{i=1}^{n}
\pi_i(a_t \mid s_t)^{w_i(s_t,z)} .
```

Each primitive expert is a Gaussian:

```latex
\pi_i(a \mid s)
=
\mathcal{N}(\mu_i(s), \Sigma_i(s)).
```

The gating network outputs state- and skill-conditioned composition weights:

```latex
w(s,z)=\mathrm{softmax}(g_\theta(s,z)/T),
```

where:

- `n` = number of primitive heads
- `z` = DADS skill latent, fixed for the skill horizon
- `w(s,z)` = internal meta-skill / primitive composition weight
- `T` = softmax temperature

Important distinction:

- `z` is not directly the primitive weight.
- `z` conditions the gating network.
- `w(s,z)` may change at every step as the state changes.

This gives an implicit hierarchy inside a single actor:

```latex
z \rightarrow w_t = w(s_t,z) \rightarrow a_t .
```

## Gaussian PoE Computation

For each head:

```latex
\Lambda_i(s) = \Sigma_i(s)^{-1}.
```

The final precision and covariance are:

```latex
\Lambda_z(s)
=
\sum_i w_i(s,z)\Lambda_i(s),
```

```latex
\Sigma_z(s)
=
\Lambda_z(s)^{-1}.
```

The final mean is:

```latex
\mu_z(s)
=
\Sigma_z(s)
\sum_i w_i(s,z)\Lambda_i(s)\mu_i(s).
```

Then sample:

```latex
a \sim \mathcal{N}(\mu_z(s), \Sigma_z(s)).
```

For diagonal Gaussian heads, implement all operations elementwise.

Recommended numerical safeguards:

- clamp per-head `log_std`
- clamp final `log_std`
- prevent final variance from becoming too small
- start with normalized weights, i.e. `sum_i w_i = 1`
- avoid negative weights

## Losses

Keep the original DADS RL loss using the DADS intrinsic reward.

The full actor-side objective should be:

```latex
\mathcal{L}
=
\mathcal{L}_{DADS}
+
\lambda_h \mathcal{L}_{head}
+
\lambda_r \mathcal{L}_{resp}
+
\lambda_e \mathcal{L}_{entropy}.
```

The last three terms are regularizers.

## 1. Head Diversity Regularizer

Purpose:

> prevent primitive Gaussian heads from collapsing to identical local action distributions.

Use a pairwise symmetric KL margin between expert heads:

```latex
D_{\mathrm{symKL}}(\pi_i,\pi_j)
=
\frac{1}{2}
\left[
D_{\mathrm{KL}}(\pi_i \| \pi_j)
+
D_{\mathrm{KL}}(\pi_j \| \pi_i)
\right].
```

Regularizer:

```latex
\mathcal{L}_{head}
=
\frac{1}{|\mathcal{P}|}
\sum_{i<j}
\max(0, m - D_{\mathrm{symKL}}(\pi_i,\pi_j)).
```

Interpretation:

- Do not force orthogonality.
- Only require heads to be distinguishable up to a margin.
- Keep this coefficient small at first.

Initial values to try:

- `m = 0.5` or `1.0`
- `lambda_h = 1e-3` to `1e-2`

## 2. Head Responsibility Balance

Purpose:

> prevent the gating network from using only one or two heads.

Compute average responsibility over a batch:

```latex
\bar{w}_i
=
\mathbb{E}_{s,z}[w_i(s,z)].
```

Regularizer:

```latex
\mathcal{L}_{resp}
=
\sum_i
\left(
\bar{w}_i - \frac{1}{n}
\right)^2.
```

Interpretation:

- This does not force every sample to use all heads.
- It only encourages all heads to be used across the batch.

Initial values:

- `lambda_r = 1e-2` to `1e-1`

## 3. Entropy Control

PoE can collapse variance because precisions add:

```latex
\Lambda_z = \sum_i w_i \Lambda_i .
```

Use the original SAC/PPO entropy term if available.

If DADS implementation does not already stabilize policy entropy, add:

- final Gaussian entropy bonus, or
- lower bound on final standard deviation, or
- target entropy penalty.

Avoid letting the final PoE distribution become nearly deterministic too early.

## What To Keep From DADS

The DADS objective should still make different fixed `z` values induce different predictable dynamics.

Conceptually:

```latex
z = \text{long-horizon exploration intent}.
```

The new actor structure changes how this intent is executed:

```latex
w_t = w(s_t,z) = \text{state-dependent primitive composition}.
```

Thus, the method should still optimize the original DADS skill diversity / predictability reward, but the policy must realize this diversity through reusable expert composition.

## Minimal Implementation Steps

1. Locate the DADS actor class.

2. Replace the single Gaussian output with:

   - `n` Gaussian expert heads producing `mu_i(s), log_std_i(s)`
   - one gating network producing `w(s,z)`

3. Implement diagonal Gaussian PoE:

   - precision = inverse variance
   - final precision = weighted sum of precisions
   - final variance = inverse final precision
   - final mean = precision-weighted average of expert means

4. Ensure the actor returns:

   - sampled action
   - final log-prob under the PoE Gaussian
   - final entropy if needed
   - optional diagnostics: `w`, per-head means/stds

5. Add `L_head` and `L_resp` to the actor loss.

6. Log diagnostics:

   - average `w_i`
   - entropy of `w`
   - pairwise symmetric KL between heads
   - final policy entropy
   - per-head std
   - DADS intrinsic reward
   - state coverage / skill diversity metric

## Important Diagnostics

Check for the following failure modes.

### Expert collapse

Symptoms:

- pairwise KL close to zero
- all head means nearly identical

Fix:

- increase `lambda_h`
- increase KL margin slightly
- initialize heads differently

### Responsibility collapse

Symptoms:

- one `w_i` dominates across nearly all states and skills
- several heads receive near-zero responsibility

Fix:

- increase `lambda_r`
- increase gating temperature
- add entropy target for `w`

### PoE entropy collapse

Symptoms:

- final log-std becomes too small
- exploration dies
- DADS intrinsic reward collapses

Fix:

- stronger entropy bonus
- clamp final log-std
- reduce precision scale
- increase per-head minimum std

### Outcome-code degeneracy

Symptoms:

- different `z` values produce different occupancy
- but internal `w(s,z)` is almost constant or one-hot per `z`
- heads are not reused across multiple skills

Fix:

- inspect `w(s,z)` across different `z`
- encourage reuse by responsibility balance
- avoid making `w` too sparse too early

## Ablations

Run at least these:

1. Original DADS actor
2. PoE actor without head/responsibility regularizers
3. PoE actor + head KL only
4. PoE actor + responsibility balance only
5. PoE actor + both regularizers
6. MoE version instead of PoE, if easy

The key question:

> Does the PoE actor preserve or improve DADS state coverage while producing reusable expert heads that are shared across many `z` values?

## Suggested Method Name

Working name:

**PoE-DADS: Product-of-Experts Skill Composition for Dynamics-Aware Skill Discovery**

Alternative:

**Meta-Action PoE Skill Discovery**

## Main Claim To Test

The method should support this claim:

> DADS learns diverse predictable dynamics for different fixed skill latents. However, a fixed latent can still entangle long-horizon exploration intent with short-horizon motor control. By parameterizing the DADS actor as a state-dependent Product-of-Experts over primitive Gaussian heads, we encourage each long-horizon skill latent to be executed through reusable internal motor components rather than as a monolithic outcome-specific policy.

