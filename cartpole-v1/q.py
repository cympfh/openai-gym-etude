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
    """Q-value Model"""

    def alpha(self, time: int) -> float:
        t = 10000
        a0 = 0.9
        return a0 if time <= t else t / time * a0

    def __init__(self,
                 verbose: bool = False):
        self.gamma = 1.0
        self.verbose = verbose
        self.time = 0

        self.model = torch.nn.Sequential(
                torch.nn.Linear(4, 1000),
                torch.nn.ReLU(),
                torch.nn.Linear(1000, 1000),
                torch.nn.Sigmoid(),
                torch.nn.Linear(1000, 2),
        ).train()
        self.opt = torch.optim.SGD(self.model.parameters(),
                                   lr=0.01,
                                   momentum=0.01,
                                   nesterov=True,
                                   )
        self.loss = torch.nn.SmoothL1Loss()

    def values(self, state: State) -> torch.tensor:
        x = torch.tensor(state, dtype=torch.float32)
        y = self.model(x)
        return y

    def fit(self, s, a, r, s_n, done) -> float:
        """
        Returns
        -------
        td : float
        """
        alpha = self.alpha(self.time)
        self.time += 1
        q_old = self.values(s).gather(0, torch.tensor([a]))
        q_future = 0 if done else self.values(s_n).max()
        td = (r - self.gamma * q_old + float(q_future)) * alpha
        # print(s, q_old.item(), td.item())

        loss = self.loss(td, torch.tensor(0.0))  # min td ** 2
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        for param in self.model.parameters():
            param.clamp(-1, 1)

        return float(td)

    def argmax(self, s):
        return int(self.values(s).argmax())


class Decision:
    """Decision maker"""

    def __init__(self, env: gym.Env, q: Q):
        self.env = env
        self.q = q
        self.tries = 0

    def epsilon(self):
        """epsilon-greedy"""
        return 0.1
        # self.tries += 1
        # return max(0.1, 1.0 / (1 + self.tries // 200))

    def __call__(self, state: State) -> Action:
        """choice an action"""
        # epsilon-greedy
        if random.random() < self.epsilon():
            return self.env.action_space.sample()
        return self.q.argmax(state)


@click.command()
@click.option('--episodes', '-E', type=int, default=2000)
@click.option('--verbose', '-v', is_flag=True, default=False)
@click.option('--render', is_flag=True, default=False)
def main(episodes, verbose, render):

    def run_episode(env: gym.Env,
                    dec: Decision,
                    q: Q,
                    episode: int,
                    max_iterate: int = 500) -> Tuple[float, List[float]]:
        """run an episode

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

            # end cost
            if done and time < 490:
                reward = 0

            # update
            if verbose:
                click.secho(
                        f'time={time}; '
                        f'action={action}; '
                        f'reward={reward} ', fg='yellow')

            loss = q.fit(state, action, reward, state_next, done)
            losses.append(loss)
            state = state_next

            if render:
                env.render()

            if done:
                break

        for _ in range(4):
            q.fit(state, action, reward, state_next, done)

        env.close()
        return score, losses

    env = make_env()
    q = Q(verbose=verbose)
    dec = Decision(env, q)

    if vis:
        Y = [0.0]
        X = [0.0]
        # win_loss = vis.line(Y, X, opts={'title': 'loss (average/episode)'})
        title = f'score'
        win_score = vis.line(Y, X, opts={'title': title})

    for episode in range(episodes):

        render = episode == episodes - 1

        score, losses = run_episode(env, dec, q, episode)
        loss_avg = sum(losses) / len(losses)

        if vis:
            # vis.line([loss_avg], [episode], win=win_loss, update='append')
            vis.line([score], [episode], win=win_score, update='append')

        click.secho(f'Episode:{episode: 4d} '
                    f'score={score:3.0f}, '
                    f'loss={loss_avg:.2f}')


if __name__ == '__main__':

    main()
