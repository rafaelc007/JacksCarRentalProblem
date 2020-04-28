import numpy as np


class CarRental:
    _num_of_cars = 0
    _rent_mean = None
    _return_mean = None

    def __init__(self, ren_m, ret_m):
        if ren_m > 0:
            self._rent_mean = ren_m
        else:
            raise Exception("Error setting rent mean parameter: must be > 0")
        if ret_m > 0:
            self._return_mean = ret_m
        else:
            raise Exception("Error setting return mean parameter: must be > 0")

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
    _stores = []
    _accumulated_cash = 0

    def __init__(self, store_param: list):
        """
        Create a new conglomerate of stores
        :param store_param: list of parameters for the stores dynamics in format: [ren_m, ret_m]*num_of_stores
        """
        [self._stores.append(CarRental(s_param[0], s_param[1])) for s_param in store_param]

    def rent_car(self, store_idx: int):
        rent_num = self._stores[store_idx].rent_car()
        self._accumulated_cash += (10*rent_num)

    def return_car(self, store_idx: int):
        ret_num = self._stores[store_idx].return_car()

    def _move_from_store(self, store_idx, amount):
        if self._stores[store_idx]._num_of_cars < amount:
            return 0
        else:
            self._stores[store_idx].remove_car(amount)
            return 1

    def move_car(self, from_idx: int, to_idx: int, amount: int):
        if amount < 0:
            raise Exception("Can't move negative amount!")
        elif amount == 0:
            return
        else:
            # moving cars
            if self._move_from_store(from_idx, amount):
                self._stores[to_idx].add_car(amount)
                self._accumulated_cash -= (2 * amount)
            else:
                print("Can't move {} cars from store {} to store {}!".format(amount, from_idx, to_idx))

    def __str__(self):
        ret_str = [("Store {} => cars: {}".format(num, store._num_of_cars)) for num, store in enumerate(self._stores)]
        ret_str.append("Accumulated cash: ${}.00".format(self._accumulated_cash))
        return "\n".join(ret_str)


if __name__ == "__main__":
    rent = JackRental([[3, 3], [4, 2]])
    print(rent)
