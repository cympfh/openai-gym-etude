import itertools
import math
import random
from typing import List, Tuple

import click
import gym
import numpy
import torch
import torch.nn

try:
    import visdom
    vis = visdom.Visdom(env='cartpole')
except ImportError:
    vis = None

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
                 alpha=1.0,
                 alpha_min=0.01,
                 gamma=0.98,
                 lr=0.3,
                 decay=0.001,
                 batch_size=64, memory_size=64*4):

        # hyper parameters for Q-learning
        self.decay = decay
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size

        # learning Q-value by NN
        self.model = torch.nn.Sequential(
                torch.nn.Linear(4, 8),
                torch.nn.ReLU(),
                torch.nn.Linear(8, 2),
        )
        self.loss = torch.nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=decay)

        self.env = env
        self.memory = []

    def value(self, state: State, action: Action) -> torch.tensor:
        """Q-value"""
        x = torch.tensor(state, dtype=torch.float32)
        y = self.model(x)
        return y[action]

    def __call__(self, state, action):
        return self.value(state, action)

    def update(self,
               state: State,
               action: Action,
               state_next: State,
               reward: float) -> float:
        """make training data and fit

        Returns
        -------
        loss : float
        """
        self.alpha = self.alpha * (1.0 - self.decay)
        alpha = max(self.alpha_min, self.alpha)
        gamma = self.gamma

        q_prev = self.value(state, action)
        q_next = max(float(self.value(state_next, a))
                     for a in range(self.env.action_space.n))

        q = [0, 0]
        q[action] = (1 - alpha) * q_prev + alpha * (reward + gamma * q_next)
        q[1 - action] = self.value(state, 1 - action)

        self.memory.append((state, q))

        if len(self.memory) > self.memory_size:
            self.memory = self.memory[len(self.memory) - self.memory_size:]

        if len(self.memory) >= self.batch_size:
            return self.fit()

        return 0.0

    def fit(self) -> float:
        """do learning

        Returns
        -------
        loss : float
        """
        batch = random.sample(self.memory, self.batch_size)
        X = []
        y = []
        for state, qvalues in batch:
            X.append(state)
            y.append(qvalues)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        y_pred = self.model(X)
        loss = self.loss(y_pred, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()


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


@click.command()
@click.option('--verbose', is_flag=True, default=False)
@click.option('--render', is_flag=True, default=False)
def main(verbose, render):

    def myreward(r: float, state: State, time, done) -> float:
        if done:
            return time - 100
        return 0
        # reg = abs(state[0]) + abs(state[1]) + abs(state[2]) + abs(state[3])
        # return 10.0 - reg * 10

    def episode(env: gym.Env,
                dec: Decision,
                q: Q,
                max_iterate: int = 500) -> Tuple[float, List[float]]:
        """take a episode

        Returns
        -------
        score : float
        losses : List[float]
        """
        state: State = env.reset()
        score = 0
        losses = []

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
            loss = q.update(state, action, state_next, reward)
            losses.append(loss)
            state = state_next

            if render:
                env.render()

            if done:
                break

        env.close()
        return score, losses

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
        for s in states:
            qs = [q(numpy.array(s), a) for a in range(2)]
            click.secho(f'{s} => {qs} => {numpy.argmax(qs)}', fg='cyan')

    env = make_env()
    q = Q(env)
    dec = Decision(env, q)
    max_score = 0.0

    if vis:
        Y = [0.0]
        X = [0.0]
        win_loss = vis.line(Y, X, opts={'title': 'loss (average/episode)'})
        win_score = vis.line(Y, X, opts={'title': 'score'})

    for ep in itertools.count():

        score, losses = episode(env, dec, q)

        if vis:
            loss_avg = sum(losses) / len(losses)
            vis.line([loss_avg], [ep+1], win=win_loss, update='append')
            vis.line([score], [ep+1], win=win_score, update='append')

        max_score = max(score, max_score)
        click.secho(f'Episode:{ep:3d} score={score}, max_score={max_score}')


if __name__ == '__main__':
    main()
