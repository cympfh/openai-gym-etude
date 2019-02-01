import math
import random

import click
import gym
import numpy
import torch
import torch.nn

# typing
Action = int
State = numpy.array


def make_env() -> gym.Env:
    env = gym.make('CartPole-v1')
    env.reset()
    return env


class Q:
    """Q-learning class

    can calculate Q-values by (state, action),
    and update them by (state, action, state_next, reward)
    """

    def __init__(self,
                 env: gym.Env,
                 alpha=1.0, alpha_min=0.01,
                 gamma=0.98,
                 lr=0.3, lr_min=0.001,
                 decay=0.0001,
                 batch_size=64, memory_size=64*4):
        self.env = env
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.gamma = gamma
        self.lr = lr
        self.lr_min = lr_min
        self.decay = decay
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.theta = torch.tensor(
                torch.randn(4, env.action_space.n, dtype=torch.float32) * 0.01,
                requires_grad=True)
        self.iter = 0
        self.memory = []

    def value(self, state: State, action: Action) -> torch.tensor:
        """Q-value"""
        x = torch.tensor(state, dtype=torch.float32)
        y = x @ self.theta
        return y[action]

    def __call__(self, state, action):
        return self.value(state, action)

    def update(self,
               state: State,
               action: Action,
               state_next: State,
               reward: float):
        """make training data and fit"""
        alpha = max(self.alpha_min,
                    self.alpha * (1.0 / (1.0 + self.decay * self.iter)))
        gamma = self.gamma

        q_prev = self.value(state, action)
        q_next = max(float(self.value(state_next, a))
                     for a in range(env.action_space.n))

        q = [0, 0]
        q[action] = (1 - alpha) * q_prev + alpha * (reward + gamma * q_next)
        q[1 - action] = self.value(state, 1 - action)

        self.memory.append((state, q))

        if len(self.memory) >= self.batch_size:
            self.fit()

        if len(self.memory) > self.memory_size:
            self.memory = self.memory[len(self.memory) - self.memory_size:]

    def fit(self):
        """do learning"""
        batch = random.sample(self.memory, self.batch_size)
        X = []
        y = []
        for state, qvalues in batch:
            X.append(state)
            y.append(qvalues)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        y_pred = X @ self.theta
        loss = ((y_pred - y) ** 2).mean()
        loss.backward()

        lr = max(self.lr_min,
                 self.lr * (1.0 / (1.0 + self.decay * self.iter)))
        self.iter += 1

        with torch.no_grad():
            self.theta -= self.theta.grad * lr
            self.theta.grad.zero_()


class Decision:
    """Decision maker"""

    def __init__(self, env: gym.Env, q: Q):
        self.tries = 0
        self.env = env
        self.q = q

    def epsilon(self):
        """epsilon-greedy"""
        return 1.0 / max(1, self.tries // 100)

    def __call__(self, state: State) -> Action:
        """choice an action"""
        self.tries += 1
        # epsilon-greedy
        if random.random() < self.epsilon():
            return self.env.action_space.sample()
        # softmax
        qs = [float(self.q(state, a)) for a in range(self.env.action_space.n)]
        zs = [math.exp(q) for q in qs]
        p = [z / sum(zs) for z in zs]
        return numpy.random.choice(len(qs), p=p)


def myreward(r: float, state: State, time, done) -> float:
    if done:
        return time - 400
    reg = abs(state[0]) + abs(state[1]) + abs(state[2]) + abs(state[3])
    return 10.0 - reg * 10


def episode(env: gym.Env,
            dec: Decision,
            q: Q,
            render: bool = False,
            verbose: bool = False,
            max_iterate: int = 500):
    """take a episode"""
    state: State = env.reset()
    score = 0

    for time in range(max_iterate):

        # take an action
        action = dec(state)
        state_next, reward, done, _ = env.step(action)
        score += reward

        # update
        reward = myreward(reward, state_next, time, done)
        if verbose:
            click.secho(
                    f'time={time}; '
                    f'action={action}; '
                    f'state={state} -> {state_next}; '
                    f'reward={reward} ', err=True)
        q.update(state, action, state_next, reward)
        state = state_next

        if render:
            env.render()
        if done:
            break

    env.close()
    return score


def test():
    states = [
        [0, 0, 0.1, 0.1],
        [0, 0, -0.1, -0.1],
        [0, 0, 0.2, 0.2],
        [0, 0, -0.2, -0.2],
        [2, -1, 0.1, 0.1],
        [2, -1, -0.1, -0.1],
        [-1, 1, 0.2, 0.2],
        [-1, 1, -0.2, -0.2],
    ]
    print(q.theta)
    for s in states:
        qs = [q(numpy.array(s), a) for a in range(2)]
        click.secho(f'{s} => {qs} => {numpy.argmax(qs)}', fg='cyan')


env = make_env()
q = Q(env)
dec = Decision(env, q)

max_score = 0.0
M = 1000
for ep in range(10000):
    score = episode(env, dec, q, render=(ep % M == 1), verbose=(ep % M == 1))
    max_score = max(score, max_score)
    click.secho(f'Episode:{ep:3d} score={score}, max_score={max_score}')
    if ep % 100 == 0:
        test()
