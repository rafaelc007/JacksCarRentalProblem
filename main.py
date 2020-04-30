import numpy as np


class CarRental:
    _num_of_cars = 0
    _rent_mean = None
    _return_mean = None

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
        self._num_of_cars = min(20, self._num_of_cars+n_to_add)
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
            ans += (store._num_of_cars * 21 ** idx)
        return ans

    def map_state(self, state_num):
        """
        map numeric state index to a sequence of car numbers
        :param state_num: state index
        :return: [num_of_cars_store_n, ... num_of_cars_store_1, num_of_cars_store_0]
        """
        if state_num > sum([20*21**idx for idx in range(len(self._stores))]):
            raise Exception("State value {} not defined!".format(state_num))
        store_car_num = []
        for idx in range(len(self._stores), 0, -1):
            div_num = state_num//(21**idx)
            if div_num == 0:
                store_car_num.append(state_num % 21)
            else:
                store_car_num.append(div_num)
            state_num = state_num % (21**idx)
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
                car_count[to_idx] + amount == next_car_count[to_idx]:
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
    [print("-", end="") for _ in range(20)]
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
    for idx1 in range(22):
        for idx2 in range(20):
            rent = JackRental([[3, 3, idx2], [4, 2, idx1]])
            print(rent.map_state())


def test_env_dynamics():
    rent = JackRental([[3, 3, 5], [4, 2, 5]])
    for state in range(441):
        for action in range(42):
            for n_state in range(441):
                if rent.env_dynamics(state, action, n_state):
                    print("State: {}, n_state: {}, action {} is true, reward {}".format(rent.map_state(state), rent.map_state(n_state), action, rent.env_reward(action)))


def test_pol_eval():
    rent = JackRental([[3, 3, 5], [4, 2, 5]])
    policy = np.matrix([[0.5] * 41] * 441)
    V = pol_eval(rent, policy)
    print(V)


def map_action(act_num: int):
    """
    simplified to work only for two stores (sorry)
    :param act_num:  action ID
    :return: [from_idx, to_idx, amount]
    """
    if act_num == 0:
        return (0, 0, 0)
    elif act_num < 21:
        return (0, 1, act_num % 21)
    elif act_num < 41:
        return (1, 0, (act_num % 21) + 1)
    else:
        raise Exception("Action value {} not defined!".format(act_num))


def pol_eval(env, policy, theta=1e-4):
    """
    :param policy: matrix of policy in format [n_states, n_actions]
    :param theta: threshold to stop the evaluation
    :return: state values after policy evaluation
    """
    V = np.zeros(policy.shape[0], dtype=float)
    delta = 1e100
    while delta > theta:
        delta = 0.0
        for s in range(len(V)):
            a = np.max(policy[s,:])
            new_V = sum([env.env_dynamics(s, a, n_s)*(env.env_reward(a) + 0.9*V[n_s]) for n_s in range(len(V))])
            delta = np.maximum(delta, np.abs(V[s] - new_V))
            V[s] = new_V
    return V


def policy_iteration():
    return


if __name__ == "__main__":
    rent = JackRental([[3, 3, 5], [4, 2, 5]])
    test_pol_eval()
