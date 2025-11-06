import numpy as np
from RLalgs.utils import epsilon_greedy
import random

def QLearning(env, num_episodes, gamma, lr, e):
    """
    Implement the Q-learning algorithm following the epsilon-greedy exploration.

    Inputs:
    env: OpenAI Gym environment 
            env.P: dictionary
                    P[state][action] are tuples of tuples tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    num_episodes: int
            Number of episodes of training
    gamma: float
            Discount factor.
    lr: float
            Learning rate.
    e: float
            Epsilon value used in the epsilon-greedy method.

    Outputs:
    Q: numpy.ndarray
    """
    
    # Unwrap if needed (for Gymnasium)
    if not hasattr(env, "P") and hasattr(env, "unwrapped"):
        env = env.unwrapped

    # Use new API attributes
    nS = env.observation_space.n
    nA = env.action_space.n

    Q = np.zeros((nS, nA))
    
    #TIPS: Call function epsilon_greedy without setting the seed
    #      Choose the first state of each episode randomly for exploration.
    ############################
    # YOUR CODE STARTS HERE

    for _ in range(num_episodes):
        # reset first
        _, _ = env.reset()

        # pick a random start state and force the env there
        start_state = np.random.randint(nS)
        env.s = start_state          # FrozenLake still exposes .s
        state = start_state

        done = False
        while not done:
            action = epsilon_greedy(Q[state], e)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            target = reward + (0 if done else gamma * np.max(Q[next_state]))
            Q[state, action] += lr * (target - Q[state, action])

            state = next_state

    # YOUR CODE ENDS HERE
    ############################

    return Q