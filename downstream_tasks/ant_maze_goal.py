import numpy as np

from envs.mujoco.ant_env import AntEnv


class AntMazeGoalEnv(AntEnv):
    """Ant navigation task with simple axis-aligned maze walls in xy space.

    The MuJoCo model is unchanged; maze walls are enforced by rejecting root xy
    transitions that enter blocked rectangles or leave the maze bounds. This keeps
    the task lightweight and compatible with pretrained Ant option policies.
    """

    def __init__(
            self,
            goal=(6.0, 6.0),
            start=(0.0, 0.0),
            goal_radius=0.75,
            max_path_length=400,
            bounds=(-1.5, 7.5, -1.5, 7.5),
            wall_penalty=1.0,
            success_bonus=50.0,
            progress_scale=10.0,
            **kwargs,
    ):
        self.goal = np.asarray(goal, dtype=np.float64)
        self.start = np.asarray(start, dtype=np.float64)
        self.goal_radius = float(goal_radius)
        self.max_path_length = int(max_path_length)
        self.bounds = tuple(float(v) for v in bounds)
        self.wall_penalty = float(wall_penalty)
        self.success_bonus = float(success_bonus)
        self.progress_scale = float(progress_scale)
        self._maze_step_count = 0
        self._last_distance = None
        self._last_collision = False
        self.walls = [
            (1.5, 5.5, 1.5, 2.2),
            (1.5, 2.2, 1.5, 5.5),
            (3.8, 4.5, 2.2, 6.2),
        ]
        super().__init__(task="motion", **kwargs)

    def reset_model(self):
        super().reset_model()
        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()
        qpos[0:2] = self.start + np.random.uniform(-0.15, 0.15, size=2)
        qvel[0:2] = 0.0
        self.set_state(qpos, qvel)
        self._maze_step_count = 0
        self._last_collision = False
        self._last_distance = self.distance_to_goal()
        return self._get_obs()

    def distance_to_goal(self):
        return float(np.linalg.norm(self.sim.data.qpos[:2] - self.goal))

    def in_wall(self, xy):
        x, y = float(xy[0]), float(xy[1])
        xmin, xmax, ymin, ymax = self.bounds
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return True
        return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in self.walls)

    def compute_reward(self, **kwargs):
        distance = self.distance_to_goal()
        if self._last_distance is None:
            self._last_distance = distance
        progress = self._last_distance - distance
        reward = self.progress_scale * progress - 0.05
        if self._last_collision:
            reward -= self.wall_penalty
        if distance <= self.goal_radius:
            reward += self.success_bonus
        self._last_distance = distance
        return float(reward)

    def _get_done(self):
        return self._maze_step_count >= self.max_path_length or self.distance_to_goal() <= self.goal_radius

    def step(self, action, render=False):
        qpos_before = self.sim.data.qpos.copy()
        qvel_before = self.sim.data.qvel.copy()
        dist_before = self.distance_to_goal()
        obs, reward, done, info = super().step(action, render=render)
        self._maze_step_count += 1

        self._last_collision = self.in_wall(self.sim.data.qpos[:2])
        if self._last_collision:
            qpos = self.sim.data.qpos.copy()
            qvel = self.sim.data.qvel.copy()
            qpos[0:2] = qpos_before[0:2]
            qvel[0:2] = 0.0
            self.set_state(qpos, qvel)
            self._last_distance = dist_before
            obs = self._get_obs()
            reward = -self.wall_penalty - 0.05

        distance = self.distance_to_goal()
        success = distance <= self.goal_radius
        done = self._maze_step_count >= self.max_path_length or success
        info.update({
            "goal": self.goal.copy(),
            "distance_to_goal": distance,
            "success": bool(success),
            "maze_collision": bool(self._last_collision),
            "coordinates": qpos_before[:2].copy(),
            "next_coordinates": self.sim.data.qpos[:2].copy(),
        })
        return obs, float(reward), done, info
