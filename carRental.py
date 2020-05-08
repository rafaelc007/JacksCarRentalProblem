import numpy as np

class CarRental:
    num_of_cars = 0
    _rent_mean = None
    _return_mean = None
    max_cars = 20

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
            self.num_of_cars = start_num

    def _remove_car(self, n_to_remove):
        self.num_of_cars = max(0, self.num_of_cars - n_to_remove)
        return self.num_of_cars

    def _add_car(self, n_to_add):
        self.num_of_cars = min(self.max_cars, self.num_of_cars + n_to_add)
        return self.num_of_cars

    def _rent_car(self):
        n_rent = np.random.poisson(self._rent_mean)
        car_num = self.num_of_cars
        self._remove_car(n_rent)
        return car_num - self.num_of_cars

    def _return_car(self):
        n_ret = np.random.poisson(self._return_mean)
        car_num = self.num_of_cars
        self._add_car(n_ret)
        return self.num_of_cars - car_num


if __name__ == "__main__":
    rent = CarRental(3,4)
