import numpy as np
from carRental import CarRental


N_ACTIONS = 11  # n_stores!*n_cars_max+1
MAX_MOVE_CARS = 5   # max number of cars allowed to be moved

class JackRental:
    _accumulated_cash = 0

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
        else:
            self._accumulated_cash += (10*rent_num)

    def _return_car(self, store_idx: int):
        ret_num = self._stores[store_idx]._return_car()

    def _can_move_from_store(self, store_idx, amount):
        return self._stores[store_idx].num_of_cars >= amount

    def move_car(self, from_idx: int, to_idx: int, amount: int):
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
            if self._can_move_from_store(from_idx, amount):
                self._stores[from_idx]._remove_car(amount)
                self._stores[to_idx]._add_car(amount)
                self._accumulated_cash -= (2 * amount)
            else:
                print("Can't move {} cars from store {} to store {}!".format(amount, from_idx, to_idx))

    def step(self):
        for store_idx in range(self._n_stores):
            self._rent_car(store_idx)
            self._return_car(store_idx)

    def get_state(self):
        return [self._stores[idx].num_of_cars for idx in range(self._n_stores)]

    def __str__(self):
        ret_str = [("Store {} => cars: {}".format(num, store.num_of_cars)) for num, store in enumerate(self._stores)]
        ret_str.append("Accumulated cash: ${}.00".format(self._accumulated_cash))
        return "\n".join(ret_str)