import numpy as np


N_ACTIONS = 11 # n_stores!*n_cars_max+1

class CarRental:
    _num_of_cars = 0
    _rent_mean = None
    _return_mean = None
    _max_cars = 5

    def __init__(self, ren_m, ret_m, start_num=0):   # a lot can go wrong with this initialization
        if ren_m > 0:
            self._rent_mean = ren_m
        else:
            raise Exception("Error setting rent mean parameter: must be > 0")
        if ret_m > 0:
            self._return_mean = ret_m
        else:
            raise Exception("Error setting return mean parameter: must be > 0")
        if start_num >= 0:
            self._num_of_cars = start_num

    def remove_car(self, n_to_remove):
        self._num_of_cars = max(0, self._num_of_cars-n_to_remove)
        return self._num_of_cars

    def add_car(self, n_to_add):
        self._num_of_cars = min(self._max_cars, self._num_of_cars+n_to_add)
        return self._num_of_cars

    def rent_car(self):
        n_rent = np.random.poisson(self._rent_mean)
        car_num = self._num_of_cars
        self.remove_car(n_rent)
        return car_num - self._num_of_cars

    def return_car(self):
        n_ret = np.random.poisson(self._return_mean)
        car_num = self._num_of_cars
        self._num_of_cars += n_ret
        return car_num - self._num_of_cars


class JackRental:
    _accumulated_cash = 0

    def __init__(self, store_param: list):
        """
        Create a new conglomerate of stores
        :param store_param: list of parameters for the stores dynamics in format: [ren_m, ret_m]*num_of_stores
        """
        self._stores = [CarRental(*s_param) for s_param in store_param]
        self._max_cars = self._stores[0]._max_cars
        self._n_stores = len(self._stores)
        self._n_states = (self._max_cars+1)**self._n_stores

    def rent_car(self, store_idx: int):
        rent_num = self._stores[store_idx].rent_car()
        if rent_num == 0:
            print("missed rent")
        else:
            self._accumulated_cash += (10*rent_num)

    def return_car(self, store_idx: int):
        ret_num = self._stores[store_idx].return_car()

    def _can_move_from_store(self, store_idx, amount):
        return self._stores[store_idx]._num_of_cars >= amount

    def move_car(self, from_idx: int, to_idx: int, amount: int):
        if from_idx == to_idx:
            return
        if amount < 0:
            raise Exception("Can't move negative amount!")
        elif amount == 0:
            return
        else:
            # moving cars
            if self._can_move_from_store(from_idx, amount):
                self._stores[from_idx].remove_car(amount)
                self._stores[to_idx].add_car(amount)
                self._accumulated_cash -= (2 * amount)
            else:
                print("Can't move {} cars from store {} to store {}!".format(amount, from_idx, to_idx))

    def __str__(self):
        ret_str = [("Store {} => cars: {}".format(num, store._num_of_cars)) for num, store in enumerate(self._stores)]
        ret_str.append("Accumulated cash: ${}.00".format(self._accumulated_cash))
        return "\n".join(ret_str)

    def map_state(self):
        ans = 0
        for idx, store in enumerate(self._stores):
            ans += (store._num_of_cars * (self._max_cars+1) ** idx)
        return ans

    def map_state(self, state_num):
        """
        map numeric state index to a sequence of car numbers
        :param state_num: state index
        :return: [num_of_cars_store_n, ... num_of_cars_store_1, num_of_cars_store_0]
        """
        if state_num > self._n_states:
            raise Exception("State value {} not defined!".format(state_num))
        store_car_num = []
        val_div = (self._max_cars+1)
        div_num = 0
        for idx in range(self._n_stores-1, 0, -1):
            div_num = state_num//(val_div**idx)
            store_car_num.append(div_num)
            state_num = state_num % (val_div**idx)
        store_car_num.append(state_num % val_div)
        return store_car_num

    def env_dynamics(self, state, action_idx, next_state):
        """
        return probability of achieving state s' given actual state s and performed action a.
        :param state: s
        :param action_idx: a
        :param next_state: s'
        :return: p(s'|s,a)
        """
        car_count = self.map_state(state)
        next_car_count = self.map_state(next_state)
        from_idx, to_idx, amount = map_action(action_idx)
        if action_idx == 0:
            if car_count == next_car_count:
                return 1
            return 0
        elif car_count[from_idx] - amount == next_car_count[from_idx] and \
                np.min([car_count[to_idx] + amount, 5]) == next_car_count[to_idx]:
                return 1
        return 0

    def env_reward(self, action_idx):
        _, _, amount = map_action(action_idx)
        return -2*amount


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


def map_action(act_num: int):
    """
    simplified to work only for two stores (sorry)
    :param act_num:  action ID
    :return: [from_idx, to_idx, amount]
    """
    half_act = N_ACTIONS//2+1  # actually n_of stores
    if act_num == 0:
        return (0, 0, 0)
    elif act_num < half_act:
        return (0, 1, act_num)
    elif act_num < N_ACTIONS:
        return (1, 0, (act_num % half_act) + 1)
    else:
        raise Exception("Action value {} not defined!".format(act_num))


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
