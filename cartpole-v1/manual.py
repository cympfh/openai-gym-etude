import gym


env = gym.make('CartPole-v1')
env = gym.wrappers.Monitor(env, 'videos/', force=True)

obs = env.reset()
print(obs)
env.render()

reward_sum = 0

for t in range(1000):

    # make action
    x, x_dot, theta, theta_dot = obs
    action = 0 if theta + theta_dot < 0 else 1

    # take action
    obs, reward, done, info = env.step(action)
    reward_sum += reward
    print(t, action, obs, reward, info)
    env.render()

    if done:
        print(f'DONE, reward={reward_sum}')
        env.close()
        exit()
