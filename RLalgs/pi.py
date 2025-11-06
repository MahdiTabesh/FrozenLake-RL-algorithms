import numpy as np
from RLalgs.utils import action_evaluation

def policy_iteration(env, gamma, max_iteration, theta):
    """
    Implement Policy iteration algorithm.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    P[state][action] is list of tuples. Each tuple contains probability, nextstate, reward, terminal
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
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
    """

    #V = np.zeros(env.nS)
    #policy = np.zeros(env.nS, dtype = np.int32)
    
    if not hasattr(env, "P") and hasattr(env, "unwrapped"):
        env = env.unwrapped
        
    # Gymnasium / newer Gym: use spaces
    nS = env.observation_space.n
    nA = env.action_space.n
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=np.int32)

    policy_stable = False
    numIterations = 0
    
    while not policy_stable and numIterations < max_iteration:
        #Implement it with function policy_evaluation and policy_improvement
        ############################
        # YOUR CODE STARTS HERE
        
        # 1. Policy Evaluation: compute V for current policy
        V = policy_evaluation(env, policy, gamma, theta)
        
        # 2. Policy Improvement: get new policy from V
        policy, policy_stable = policy_improvement(env, V, policy, gamma)

        # YOUR CODE ENDS HERE
        ############################
        numIterations += 1
        
    return V, policy, numIterations


def policy_evaluation(env, policy, gamma, theta):
    """
    Evaluate the value function from a given policy.

    Inputs:
    env: OpenAI Gym environment.
            env.P: dictionary
                    
            env.nS: int
                    number of states
            env.nA: int
                    number of actions

    gamma: float
            Discount factor.
    policy: numpy.ndarray
            The policy to evaluate. Maps states to actions.
    theta: float
            The threshold of convergence.
    
    Outputs:
    V: numpy.ndarray
            The value function from the given policy.
    """
    ############################
    # YOUR CODE STARTS HERE
    
    if not hasattr(env, "P") and hasattr(env, "unwrapped"):
        env = env.unwrapped
        
    nS = env.observation_space.n
    nA = env.action_space.n
    V = np.zeros(nS)
    #V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(nS):
            a = policy[s]
            # Compute one-step lookahead for action a
            v_s = 0.0
            for (prob, nextstate, reward, terminal) in env.P[s][a]:
                if terminal:
                    v_s += prob * reward
                else:
                    v_s += prob * (reward + gamma * V[nextstate])
            # Update delta for convergence check
            delta = max(delta, abs(v_s - V[s]))
            V[s] = v_s
        if delta < theta:
            break

    # YOUR CODE ENDS HERE
    ############################

    return V


def policy_improvement(env, value_from_policy, policy, gamma):
    """
    Given the value function from policy, improve the policy.

    Inputs:
    env: OpenAI Gym environment
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

    value_from_policy: numpy.ndarray
            The value calculated from the policy
    policy: numpy.ndarray
            The previous policy.
    gamma: float
            Discount factor.

    Outputs:
    new policy: numpy.ndarray
            An array of integers. Each integer is the optimal action to take
            in that state according to the environment dynamics and the
            given value function.
    policy_stable: boolean
            True if the "optimal" policy is found, otherwise false
    """
    ############################
    # YOUR CODE STARTS HERE
    # unwrap for Gymnasium
    if not hasattr(env, "P") and hasattr(env, "unwrapped"):
        env = env.unwrapped

    nS = env.observation_space.n
    nA = env.action_space.n

    policy_stable = True

    for s in range(nS):
        q_best = -float('inf')
        best_a = None

        for a in range(nA):
            q_sa = 0.0
            for (prob, nextstate, reward, terminal) in env.P[s][a]:
                if terminal:
                    q_sa += prob * reward
                else:
                    q_sa += prob * (reward + gamma * value_from_policy[nextstate])
            if q_sa > q_best:
                q_best = q_sa
                best_a = a

        # update the ORIGINAL policy in place
        if best_a != policy[s]:
            policy_stable = False
        policy[s] = best_a

    # YOUR CODE ENDS HERE
    ############################

    return policy, policy_stable