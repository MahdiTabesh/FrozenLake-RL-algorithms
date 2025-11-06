import numpy as np
from RLalgs.utils import action_evaluation

def value_iteration(env, gamma, max_iteration, theta):
    """
    Implement value iteration algorithm. 

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    the transition probabilities of the environment
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    gamma: float
            Discount factor.
    max_iteration: int
            The maximum number of iterations to run before stopping.
    theta: float
            The threshold of convergence.
    
    Outputs:
    V: numpy.ndarray
    policy: numpy.ndarray
    numIterations: int
            Number of iterations
    """
    # unwrap if it's a Gymnasium wrapped env
    if not hasattr(env, "P") and hasattr(env, "unwrapped"):
        env = env.unwrapped

    nS = env.observation_space.n
    nA = env.action_space.n

    V = np.zeros(nS)
    numIterations = 0

    #Implement the loop part here
    ############################
    # YOUR CODE STARTS HERE
    
    for i in range(max_iteration):
        delta = 0
        # Update each state's value by the best action's outcome
        for s in range(nS):
            v_best = -float('inf')
            for a in range(nA):
                q_sa = 0.0
                for (prob, nextstate, reward, terminal) in env.P[s][a]:
                    if terminal:
                        q_sa += prob * reward
                    else:
                        q_sa += prob * (reward + gamma * V[nextstate])
                if q_sa > v_best:
                    v_best = q_sa
            # Track maximum change for convergence
            delta = max(delta, abs(v_best - V[s]))
            V[s] = v_best
        numIterations += 1
        if delta < theta:
            break

    # YOUR CODE ENDS HERE
    ############################
    
    #Extract the "optimal" policy from the value function
    policy = extract_policy(env, V, gamma)
    
    return V, policy, numIterations

def extract_policy(env, v, gamma):

    """ 
    Extract the optimal policy given the optimal value-function.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    P[state][action] is tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    v: numpy.ndarray
        value function
    gamma: float
        Discount factor. Number in range [0, 1)
    
    Outputs:
    policy: numpy.ndarray
    """
    
    if not hasattr(env, "P") and hasattr(env, "unwrapped"):
        env = env.unwrapped

    nS = env.observation_space.n
    nA = env.action_space.n

    policy = np.zeros(nS, dtype = np.int32)
    ############################
    # YOUR CODE STARTS HERE
    
    for s in range(nS):
        # Choose the best action for state s by one-step lookahead
        best_a = 0
        best_val = -float('inf')
        for a in range(nA):
            q_sa = 0.0
            for (prob, nextstate, reward, terminal) in env.P[s][a]:
                if terminal:
                    q_sa += prob * reward
                else:
                    q_sa += prob * (reward + gamma * v[nextstate])
            if q_sa > best_val:
                best_val = q_sa
                best_a = a
        policy[s] = best_a

    # YOUR CODE ENDS HERE
    ############################

    return policy