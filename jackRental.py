import numpy as np
from carRental import CarRental


N_ACTIONS = 11  # n_stores!*n_cars_max+1
MAX_MOVE_CARS = 5   # max number of cars allowed to be moved

class JackRental:
    _accumulated_cash = 0

    def get_Cash(self):
        return self._accumulated_cash

    def __init__(self, store_param: list):
        """
        Create a new conglomerate of stores
        :param store_param: list of parameters for the stores dynamics in format: [ren_m, ret_m]*num_of_stores
        """
        self._stores = [CarRental(*s_param) for s_param in store_param]
        self._max_cars = self._stores[0].max_cars
        self._n_stores = len(self._stores)
        self._n_states = (self._max_cars+1)**self._n_stores

    def _rent_car(self, store_idx: int):
        rent_num = self._stores[store_idx]._rent_car()
        if rent_num == 0:
            print("missed rent")
            return 0
        else:
            self._accumulated_cash += (10*rent_num)
            return rent_num

    def _return_car(self, store_idx: int):
        ret_num = self._stores[store_idx]._return_car()
        return  ret_num

    def move_car(self, from_idx: int, to_idx: int, amount: int, verbose=False):
        if from_idx == to_idx:
            return
        if amount < 0:
            raise Exception("Can't move negative amount!")
        if amount > MAX_MOVE_CARS:
            raise Exception("Can't move that amount, max is {}".format(MAX_MOVE_CARS))
        elif amount == 0:
            return
        else:
            # moving cars
            if self._stores[from_idx].num_of_cars < amount:
                amount = self._stores[from_idx].num_of_cars
            self._stores[from_idx]._remove_car(amount)
            self._stores[to_idx]._add_car(amount)
            self._accumulated_cash -= (2 * amount)
            print("moving {} from {} to {}\n".format(amount, from_idx, to_idx))

    def step(self, verbose=False):
        for store_idx in range(self._n_stores):
            no_rent = self._rent_car(store_idx)
            no_return = self._return_car(store_idx)
            if verbose:
                print("Store {} => Rent: {}, Returned {}".format(store_idx, no_rent, no_return))

    def get_state(self):
        return [self._stores[idx].num_of_cars for idx in range(self._n_stores)]

    def __str__(self):
        ret_str = [("Store {} => cars: {}".format(num, store.num_of_cars)) for num, store in enumerate(self._stores)]
        ret_str.append("Accumulated cash: ${}.00".format(self._accumulated_cash))
        return "\n".join(ret_str)