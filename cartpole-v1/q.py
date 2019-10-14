import random
from typing import List, Tuple

import click
import gym
import torch
from gnuplot import Figure, Gnuplot
from torch import nn, optim

GYM_NAME = 'CartPole-v1'
Episode = List[Tuple[torch.tensor, int, float]]  # list of (status, action, reward)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.criterion = nn.MSELoss()

    def forward(self, x) -> torch.tensor:
        """Q-value for action=1

        Q(action=0) = -Q(action=1)
        """
        x = torch.sigmoid(x)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x.flatten()

    def action(self, x, epsilon=0.02) -> int:
        if random.random() < epsilon:
            return random.randrange(2)
        q = self.forward(x)
        expq1 = float(q.exp())
        expq0 = float((-q).exp())
        pr = expq1 / (expq0 + expq1)
        if random.random() < pr:
            return 1
        return 0

    def step(self, episode: Episode,
             optimizer: optim.Optimizer,
             gamma: float = 0.9,
             verbose: bool = False,
             ) -> float:
        """Q-learning"""
        self.train(False)
        X = []
        y = []
        r_sum = 0
        G = 50  # goal steps
        if len(episode) > G:
            r_sum = G
        for s, a, r in reversed(episode):
            r_sum -= 1
            q_old = float(self.forward(s))
            if a == 1:
                q = gamma * q_old + (1 - gamma) * r_sum
            else:
                q = gamma * q_old - (1 - gamma) * r_sum
            X.append(s)
            y.append(q)

        if verbose:
            X.reverse()
            y.reverse()
            for i, (s, a, r) in enumerate(episode):
                click.secho(f"time={i:03d}: status={s}, action={a}, "
                            f"q_old={float(self.forward(s))}, q_new={float(y[i])}",
                            fg='yellow')

        X = torch.stack(X)
        y = torch.tensor(y, dtype=torch.float32)

        self.train(True)
        optimizer.zero_grad()
        output = self.forward(X)
        loss = self.criterion(output, y)
        loss.backward()
        optimizer.step()
        return float(loss)


class EpisodePool:

    def __init__(self):
        self.pool = []

    def push(self, episode: Episode):
        self.pool.append(episode)

    def pop(self):
        return self.pool[-1]

    def random_pop(self):
        return random.choice(self.pool)


def make_episode(net: Net, env: gym.Env,
                 render=False,
                 max_iterate=500
                 ) -> Tuple[float, Episode]:
    """make an episode

    Returns
    -------
    sum_of_reward: float
    episode
    """
    status = torch.tensor(env.reset(), dtype=torch.float32)
    reward_sum = 0
    episode = []

    for t in range(max_iterate):
        net.train(False)
        action = net.action(status)
        obs, reward, done, info = env.step(action)

        status = torch.tensor(obs, dtype=torch.float32)
        reward_sum += reward
        episode.append((status, action, reward))

        if render:
            env.render()

        if done:
            break

    env.close()
    return reward_sum, episode


def make_great_net(try_times: int = 1000) -> Net:
    """Model Agnostic Learning

    returns the great (randomly generated) Net which got the max score
    """
    net = Net()
    env = gym.make(GYM_NAME)
    score = make_episode(net, env, render=False)[0]
    for _ in range(try_times):
        net2 = Net()
        score2 = make_episode(net2, env, render=False)[0]
        if score2 > score:
            score = score2
            net = net2
    click.secho(f"Model Agnostic: score={score}", fg='yellow')
    return net


@click.command()
@click.option('--epochs', '-E', type=int, default=10)
@click.option('--agnostic/--no-agnostic', default=True)
@click.option('--render/--no-render', default=False)
@click.option('--verbose', '-v', is_flag=True, default=False)
def main(epochs, agnostic, render, verbose):
    """Run and learn episodes"""
    env = gym.make(GYM_NAME)

    if render:
        env = gym.wrappers.Monitor(env, 'videos/', force=True)

    if agnostic:
        net = make_great_net()
    else:
        net = Net()

    optimizer = optim.SGD(net.parameters(), lr=0.01)
    pool = EpisodePool()
    result = []
    for i in range(epochs):
        reward, episode = make_episode(net, env, render=render)
        loss = net.step(episode, optimizer, verbose=verbose)
        click.secho(f"Episode #{i + 1:04d}: reward={reward}, loss={loss}", fg='blue')
        pool.push(episode)
        for _ in range(2):
            net.step(pool.random_pop(), optimizer)
        result.append(reward)

    with Gnuplot() as g:
        g.set('terminal', 'pngcairo')
        g.set('output', '"out.png"')
        g.set('title', '"CartPole-v0"')
        g.set('xlabel', '"episode"')
        g.set('ylabel', '"steps"')
        g.var('$dat', result)
        g.plot(Figure('$dat', _with='lp', title=None))


if __name__ == '__main__':
    main()
