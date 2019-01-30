import click
import gym


def episode(env, render=False, verbose=False, max_iterate=500):

    obs = env.reset()
    reward_sum = 0

    for t in range(max_iterate):

        # make action
        x, x_dot, theta, theta_dot = obs
        action = 0 if theta + 0.1 * theta_dot < 0 else 1

        # take action
        obs, reward, done, info = env.step(action)
        reward_sum += reward

        if verbose:
            click.secho(
                    f'time={t} '
                    f'action={action} '
                    f'observation={obs} '
                    f'reward={reward} ', err=True)

        if render:
            env.render()

        if done:
            break

    env.close()
    return reward_sum


@click.group()
def main():
    pass


@main.command()
def debug():
    """Run 1 episode with rendering and saving video"""
    env = gym.make('CartPole-v1')
    env = gym.wrappers.Monitor(env, 'videos/', force=True)
    reward = episode(env, render=True, verbose=True)
    print(f'Reward: {reward}')


@main.command()
def test():
    """Run 100 episodes, then report results"""
    env = gym.make('CartPole-v1')

    results = []
    for _ in range(100):
        results.append(episode(env, render=False, verbose=False))

    print(f'average={sum(results) / len(results)} '
          f'max={max(results)} '
          f'min={min(results)}')


if __name__ == '__main__':
    main()
