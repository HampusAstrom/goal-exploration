from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize
import numpy as np

REWARD_MODE_BINARY = 'binary' # return 1 when pendulum within allowed speed and angle limit over balance_counter time steps
REWARD_MODE_SPARSE = 'sparse' # return continuous speed reward only within speed and range limit

class SparsePendulumEnv(PendulumEnv):
    balance_counter = 0

    def __init__(self, max_speed, max_torque, reward_angle_limit, reward_speed_limit, balance_counter, reward_mode):
        super().__init__()
        self.max_speed= max_speed
        self.max_torque= max_torque
        self.reward_speed_limit = reward_speed_limit
        self.reward_angle_limit = reward_angle_limit
        self.balance_counter = balance_counter
        self.reward_mode = reward_mode

    def reward_binary(self, th, thdot, u):
        angle = np.degrees(angle_normalize(th))
        done = False
        reward = 0

        if self.check_angle_speed_limit(angle, thdot):
            SparsePendulumEnv.balance_counter += 1
            if SparsePendulumEnv.balance_counter >= self.balance_counter:
                SparsePendulumEnv.balance_counter = 0
                reward = 1
                done = True

        return reward, done

    def reward_sparse(self, th, thdot, u):
        angle = np.degrees(angle_normalize(th))
        done = False
        reward = 0

        if self.check_angle_speed_limit(angle, thdot):
            reward = self.max_speed - (np.absolute(thdot) / 6.0)

        return reward, done

    def check_angle(self, angle):
        return (angle >= -self.reward_angle_limit) and (angle <= self.reward_angle_limit)

    def check_speed(self, thdot):
        return (thdot >= -self.reward_speed_limit) and (thdot <= self.reward_speed_limit)

    def check_angle_speed_limit(self, angle, thdot):
        return self.check_angle(angle) and self.check_speed(thdot)

    def step(self, u):
        th, thdot = self.state # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        done = False
        reward = 0

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering

        if self.reward_mode == REWARD_MODE_BINARY:
            reward, done = self.reward_binary(th, thdot, u)
        elif self.reward_mode == REWARD_MODE_SPARSE:
            reward, done = self.reward_sparse(th, thdot, u)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        obs = self._get_obs()

        return obs, reward, done, {}
