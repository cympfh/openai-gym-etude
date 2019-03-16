import click
import gym
import numpy

env = gym.make('Copy-v0')
aspace = env.action_space

# sizeof state space
S = 5 + 1

# sizeof action space
I = 2
J = 2
K = 5


def initial_qtable():
    return numpy.random.rand(S, I, J, K) / 1000.0


def strOfState(state) -> str:
    if state == K:
        return 'NUL'
    return chr(65 + state)


def strOfAction(action) -> str:
    move, write, pred = action
    d = ['left', 'right'][move]
    if write:
        a = strOfState(pred)
        return f"Move to {d} and write '{a}'"
    else:
        return f"Move to {d}"


@click.command()
@click.option('--epochs', type=int, default=10000)
@click.option('--alpha', type=float, default=0.9)
@click.option('--gamma', type=float, default=0.5)
@click.option('--verbose', '-V', is_flag=True)
def main(epochs, alpha, gamma, verbose):
    """Copy-v0 Q-learning

    run `epochs` episodes"""
    qtable = initial_qtable()

    for _epi in range(epochs):

        print(f'=== Episode {_epi:03d} ===========')
        total_r = 0.0

        s = env.reset()
        env.render()
        history = []

        # an episode
        while True:

            # max action
            cs = [(qtable[s, i, j, k], i, j, k)
                  for i in range(I)
                  for j in range(J)
                  for k in range(K)]
            cs.sort(key=lambda item: -item[0])
            _score, i_prev, j_prev, k_prev = cs[0]
            s_next, r, done, _ = env.step((i_prev, j_prev, k_prev))

            total_r += r

            # max action (next)
            ds = [(qtable[s_next, i, j, k], i, j, k)
                  for i in range(I)
                  for j in range(J)
                  for k in range(K)]
            ds.sort(key=lambda item: -item[0])
            _, i_next, j_next, k_next = ds[0]

            history.append((s, i_prev, j_prev, k_prev, r, s_next, i_next, j_next, k_next))
            s = s_next

            if done:
                env.render()
                break

        # Q updating (on reversed history)
        if verbose:
            print('Updating...')
            for s, i_prev, j_prev, k_prev, r, s_next, i_next, j_next, k_next in history:
                print(f'Tape {strOfState(s)}, {strOfAction((i_prev, j_prev, k_prev))} -> '
                      f'Reward {r}; {qtable[s, i_prev, j_prev, k_prev]}')

        for s, i_prev, j_prev, k_prev, r, s_next, i_next, j_next, k_next in reversed(history):
            qtable[s, i_prev, j_prev, k_prev] *= 1.0 - alpha
            qtable[s, i_prev, j_prev, k_prev] += alpha * (gamma * qtable[s_next, i_next, j_next, k_next] + r)

        if verbose:
            print('Updated')
            for s, i_prev, j_prev, k_prev, r, s_next, i_next, j_next, k_next in history:
                print(f'Tape {strOfState(s)}, {strOfAction((i_prev, j_prev, k_prev))} -> '
                      f'Reward {r}; {qtable[s, i_prev, j_prev, k_prev]}')

        print(f'score {total_r}')


if __name__ == '__main__':
    main()
