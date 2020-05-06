import numpy as np
from jackRental import JackRental, N_ACTIONS


def test_rent():
    rent = JackRental([[3, 3, 0], [4, 2, 5]])
    routines = [rent.rent_car, rent.return_car, rent.move_car]
    print("Start value")
    print(rent)
    [print("-", end="") for _ in range(rent._max_cars)]
    print("")
    for idx, routine in enumerate(routines):
        for st1 in range(2):
            if idx < 2:
                routine(st1)
                print(routine)
                print(rent)
            else:
                for st2 in range(2):
                    for amount in range(5):
                        if st1 != st2:
                            routine(st1, st2, amount)
                            print(routine)
                            print(rent)


def test_map_state():
    rent = JackRental([[3, 3, 0], [4, 2, 5]])
    for idx1 in range(rent._max_cars):
        for idx2 in range(rent._max_cars):
            rent = JackRental([[3, 3, idx2], [4, 2, idx1]])
            print(rent.map_state())


def test_map_state2():
    rent = JackRental([[3, 3, 0], [4, 2, 5]])
    for idx in range(rent._n_states):
        print("state {} is {}".format(idx, rent.map_state(idx)))


def randargmax(b, **kw):
    """ a random tie-breaking argmax"""
    b = np.array(b)
    return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)


def test_env_dynamics():
    rent = JackRental([[3, 3, 5], [4, 2, 5]])
    for state in range(rent._n_states):
        for action in range(N_ACTIONS):
            for n_state in range(rent._n_states):
                if rent.env_dynamics(state, action, n_state):
                    print("State: {}, n_state: {}, action {} is true, reward {}".format(rent.map_state(state), rent.map_state(n_state), action, rent.env_reward(action)))


def test_pol_eval():
    rent = JackRental([[3, 3, 5], [4, 2, 5]])
    policy = np.random.randint(0, N_ACTIONS, rent._n_states)
    V = pol_eval(rent, policy, theta=1e-1)
    print(V)


def pol_eval(env, policy, theta=1e-3, V=None):
    """
    :param env: environment object
    :param policy: array of policy in format [n_states]
    :param theta: threshold to stop the evaluation
    :param V: pre-loaded value function
    :return: state values after policy evaluation
    """
    if V is None:
        V = np.zeros(env._n_states, dtype=float)
    delta = 1e100
    while delta > theta:
        delta = 0.0
        for s in range(len(V)):
            a = policy[s]
            new_V = sum([env.env_dynamics(s, a, n_s)*(env.env_reward(a) + 0.9*V[n_s]) for n_s in range(len(V))])
            delta = np.maximum(delta, np.abs(V[s] - new_V))
            # print("delta: ", delta)
            V[s] = new_V
    return V


def policy_iteration(env):
    policy = np.random.randint(0, N_ACTIONS, env._n_states)
    V = np.zeros(len(policy), dtype=float)
    Q = [0.0]*N_ACTIONS
    policy_instable = True
    while policy_instable:
        V = pol_eval(env, policy, V=V)

        for s in range(len(V)):
            old_action = policy[s]
            for a in range(N_ACTIONS):
                Q[a] = sum([env.env_dynamics(s, a, n_s)*(env.env_reward(a) + 0.9*V[n_s]) for n_s in range(len(V))])
            policy[s] = randargmax(Q)
            if old_action == policy[s] : policy_instable = False
    return policy, V


if __name__ == "__main__":
    rent = JackRental([[3, 3, 5], [4, 2, 5]])
    pol, V = policy_iteration(rent)
    print(pol)
    [print("-", end="")]*11
    print("\n", V)
    # test_pol_eval()
