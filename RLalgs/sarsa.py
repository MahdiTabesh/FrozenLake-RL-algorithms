import numpy as np
from RLalgs.utils import epsilon_greedy
import random

def SARSA(env, num_episodes, gamma, lr, e):
    """
    Implement the SARSA algorithm following epsilon-greedy exploration.

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
            State-action values
    """
    
    #TIPS: Call function epsilon_greedy without setting the seed
    #      Choose the first state of each episode randomly for exploration.
    ############################
    # YOUR CODE STARTS HERE
    
    if not hasattr(env, "P") and hasattr(env, "unwrapped"):
        env = env.unwrapped

    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA))  # initialize Q-table with zeros

    for _ in range(num_episodes):
        # Reset environment (Gymnasium's reset returns (obs, info) tuple)
        reset_out = env.reset()
        if isinstance(reset_out, tuple):
            state, _ = reset_out
        else:
            state = reset_out

        # Choose a random non-terminal start state for exploring starts (avoid terminal starts)
        if hasattr(env, "P"):
            non_terminal_states = []
            for s in range(nS):
                # State is non-terminal if any action leads to a non-terminal next state
                all_done = True
                for a in range(nA):
                    for (prob, next_s, reward, done) in env.P[s][a]:
                        if prob > 0 and not done:
                            all_done = False
                            break
                    if not all_done:
                        break
                if not all_done:
                    non_terminal_states.append(s)
            # Pick a random non-terminal state to start the episode
            state = random.choice(non_terminal_states)
            env.unwrapped.s = state

        # Initial action selection using epsilon-greedy (no fixed seed for randomness)
        action = epsilon_greedy(Q[state], e)
        done = False

        # Loop until episode ends
        while not done:
            # Take action and observe outcome (Gymnasium's step returns 5 values)
            step_out = env.step(action)
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_out

            # Update Q-value for the current state-action pair
            if done:
                # Terminal state reached: update towards reward (no next state to consider)
                Q[state, action] += lr * (reward - Q[state, action])
                break

            # Choose next action using epsilon-greedy (on-policy for SARSA)
            next_action = epsilon_greedy(Q[next_state], e)
            # Compute SARSA target using the next state-action Q-value
            target = reward + gamma * Q[next_state, next_action]
            # Update Q towards the target for non-terminal transition
            Q[state, action] += lr * (target - Q[state, action])

            # Move to next state and continue the episode
            state = next_state
            action = next_action

    # YOUR CODE ENDS HERE
    ############################

    return Q