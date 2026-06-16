# Ant Heading Counterfactual Test

## Goal

Evaluate whether learned USD skills represent **context-invariant local behaviors** or merely **global displacement directions**.

A skill is considered local/context-invariant if executing the same skill from different initial headings produces consistent motion in the agent's body frame.

## Target Methods

Run the diagnostic on existing skill-conditioned policies:

- SUSD
- METRA
- LSD
- CSD
- Optional: DIAYN, DUSDi

No algorithm modification is required.

## Environment

Use the existing `Ant` environment from the SUSD/METRA codebase.

Do **not** use AntMaze for the first diagnostic.

Required modification:

- Add a reset wrapper that can set the initial torso yaw.
- Prefer training/evaluating all methods with yaw-randomized resets to avoid OOD-reset criticism.

Initial heading set:

```python
headings = [0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
```

## Core Question

For a fixed skill `z`, does the rollout remain consistent after aligning trajectories to the agent's initial body frame?

If yes, the skill behaves like a local primitive.

If no, the skill is entangled with global context/coordinate frame.

## Protocol

### 1. Train Or Load Policies

For each method, obtain a frozen skill-conditioned policy:

```python
pi(a | s, z)
```

Use the same training budget and environment settings across methods.

Recommended:

- Train with yaw-randomized initial states.
- Keep all other initial state components fixed or sampled from the method's standard reset distribution.

### 2. Select Skills

Use a fixed set of skill vectors.

For continuous 2D skill spaces:

```python
num_skills = 16
skills = [
    np.array([np.cos(2 * np.pi * k / num_skills),
              np.sin(2 * np.pi * k / num_skills)])
    for k in range(num_skills)
]
```

For higher-dimensional skill spaces:

- Either sample `num_skills` unit-norm vectors from the skill prior.
- Or vary one skill factor at a time if the method uses factorized skills.

Use the exact same sampled skills across headings and seeds.

### 3. Counterfactual Rollouts

For each method, seed, skill, and initial heading:

```python
for method in methods:
    for seed in seeds:
        policy = load_policy(method, seed)

        for z in skills:
            for theta in headings:
                s0 = reset_ant_with_heading(theta)
                rollout policy pi(a | s, z) for H steps
                record root xy position p_t
                record initial xy position p_0
                record initial heading theta
```

Recommended rollout horizon:

```python
H = 100
```

Use deterministic policy evaluation if available.

Repeat each `(z, theta)` rollout multiple times if the policy/environment is stochastic.

```python
num_eval_rollouts = 5
```

### 4. Trajectory Representations

For every rollout, compute the start-relative trajectory in world coordinates:

```python
tau_world[t] = p_t - p_0
```

Then compute the body-aligned trajectory:

```python
R_minus_theta = np.array([
    [np.cos(-theta), -np.sin(-theta)],
    [np.sin(-theta),  np.cos(-theta)]
])

tau_body[t] = R_minus_theta @ (p_t - p_0)
```

Use only root xy position for the first version.

Optional additional local features:

```python
delta_heading[t] = yaw_t - yaw_0
forward_displacement = tau_body[:, 0]
lateral_displacement = tau_body[:, 1]
```

## Metrics

### Body-Frame Variance

Measures whether the same skill produces consistent local/body-frame behavior across headings.

```python
BodyVar(z) = mean_t trace(cov_theta(tau_body[z, theta, t]))
```

Lower is better for local primitive behavior.

### World-Frame Variance

Measures whether the same skill produces consistent global displacement across headings.

```python
WorldVar(z) = mean_t trace(cov_theta(tau_world[z, theta, t]))
```

Lower means the skill is tied to global direction.

### Locality Score

```python
LocalityScore(z) = WorldVar(z) / (BodyVar(z) + eps)
```

Interpretation:

- High `LocalityScore`: body-frame consistent, world-frame rotated by heading.
- Low `LocalityScore`: world-frame consistent, body-frame inconsistent.

Aggregate over skills:

```python
LocalityScore = mean_z LocalityScore(z)
BodyVar = mean_z BodyVar(z)
WorldVar = mean_z WorldVar(z)
```

Use bootstrap confidence intervals or report mean/std over seeds.

## Expected Outcomes

### Context-Invariant Local Skill

For the same skill `z`:

- Body-aligned trajectories overlap across headings.
- World-frame trajectories rotate according to initial heading.
- `BodyVar` is low.
- `WorldVar` is high.
- `LocalityScore` is high.

### Global Direction Skill

For the same skill `z`:

- World-frame trajectories overlap across headings.
- Body-aligned trajectories differ across headings.
- `WorldVar` is low.
- `BodyVar` is high.
- `LocalityScore` is low.

This indicates skill-context entanglement.

## Figures

For each method, show a small grid of representative skills.

For each selected skill `z`, plot:

1. World-frame rollout overlay
2. Body-aligned rollout overlay

Use one color per initial heading.

Recommended figure layout:

```text
Method: METRA

Skill z_0
[World-frame trajectories] [Body-aligned trajectories]

Skill z_4
[World-frame trajectories] [Body-aligned trajectories]
```

A strong failure case is:

- World-frame plot: trajectories overlap.
- Body-frame plot: trajectories spread apart.

This visually shows that the skill encodes global displacement rather than reusable local behavior.

## Minimal Implementation Checklist

- [ ] Add `reset_ant_with_heading(theta)` wrapper.
- [ ] Load frozen policies for SUSD/METRA/baselines.
- [ ] Define fixed skill set.
- [ ] Run rollouts for all `(method, seed, skill, heading)`.
- [ ] Save `tau_world` and `tau_body`.
- [ ] Compute `BodyVar`, `WorldVar`, `LocalityScore`.
- [ ] Plot world-frame and body-aligned trajectories.
- [ ] Report mean/std across seeds.

## Claim Supported By This Diagnostic

Existing USD methods can learn diverse and useful skills, but their skill variables may remain entangled with the global coordinate frame or initial context.

This experiment directly tests whether a learned skill denotes a reusable local transition operator.

Suggested paper wording:

> We evaluate whether discovered skills correspond to context-invariant local behaviors by executing the same skill under counterfactual initial headings. If a skill represents a reusable local primitive, its induced trajectory should be consistent in the agent-centric frame. In contrast, skills that encode global displacement directions will remain consistent in the world frame but vary in the body frame.
