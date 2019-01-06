#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import environment_discrete
import numpy as np

if __name__ == "__main__":
    # pylint: disable=C0103

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1", type=str, help="Name of the environment.")
    parser.add_argument("--episodes", default=1000, type=int, help="Episodes in a batch.")
    parser.add_argument("--max_steps", default=500, type=int, help="Maximum number of steps.")
    parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")

    parser.add_argument("--epsilon", default=0.5, type=float, help="Epsilon.")
    parser.add_argument("--epsilon_final", default=0.01, type=float, help="Epsilon decay rate.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discounting factor.")
    args = parser.parse_args()

    # Create the environment
    env = environment_discrete.EnvironmentDiscrete(args.env)

    # Create Q, C and other variables
    Q = np.zeros([env.states, env.actions])
    C = np.zeros([env.states, env.actions]) # pocet kolik jich bylo

    epsilon = args.epsilon
    episode_rewards, episode_lengths = [], []

    for episode in range(args.episodes):        
        # Perform episode
        state = env.reset()        
        states, actions, rewards, total_reward = [], [], [], 0
        for t in range(args.max_steps):
            if args.render_each and episode > 0 and episode % args.render_each == 0:
                env.render()

            # akce jsou od 0 do env.actions
            # stavy jsou od 0 do env.states

            # Compute action using epsilon-greedy policy
            action = 0
            random = np.random.random()
            if random < 1 - epsilon:                
                action = np.argmax(Q[state])
            else:
                action = np.random.choice(env.actions)
            
            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            if done:
                break

        # Sum and discount rewards
        gamma = args.gamma        
        for i in range(len(states)):            
            # g 'sum' rewards[i:]
            #g = np.sum(rewards[i:]) * gamma

            g = 0            
            state = states[i]
            action = actions[i]    
            # g by se mělo počítat tak, aby se to celé provedlo v lineárním čase
            # (tj. jako částečné součty od konce posloupnosti).
            # Update Q and C
            if C[state, action] == 0:                
                Q[state, action] = g
            else:                                       
                for j in range(i, len(states)):
                    g += rewards[j] * np.power(gamma, j-1)
                Q[state, action] = Q[state, action] + (1 / (C[state, action] + 1) * (g - Q[state, action]))

            C[state, action] += 1

            '''
            C(s,a) += 1
            Q(s,a) = Q(s,a) + 1/C(s,a)*(G_t - Q(s,a))
            '''

        episode_rewards.append(total_reward)
        episode_lengths.append(t)
        if len(episode_rewards) % 10 == 0:
            print("Episode {}, mean 100-episode reward {}, mean 100-episode length {}, epsilon {}.".format(
                episode + 1, np.mean(episode_rewards[-100:]), np.mean(episode_lengths[-100:]), epsilon))

        if args.epsilon_final:
            epsilon = np.exp(np.interp(episode + 1, [0, args.episodes], [np.log(args.epsilon), np.log(args.epsilon_final)]))
