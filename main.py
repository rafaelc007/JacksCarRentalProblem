import matplotlib.pyplot as plt
from jackRental import JackRental, N_ACTIONS
import numpy as np

def test_rent():
    rent = JackRental([[3, 3, 0], [4, 2, 5]])
    routines = [rent._rent_car, rent._return_car, rent.move_car]
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

def trans_action(action):
    if action > 0:
        return 0, 1, action
    elif action < 0:
        return 1, 0, -action
    else:
        return 0, 0, 0

def run_car_rent(policy):
    rent = JackRental([[1, 2, 5], [2, 1, 5]])
    for idx in range(10):
        rent.step(True)
        print("rent state:")
        print(rent)
        state = rent.get_state()
        action = policy[state[0]][state[1]]
        str1, str2, num = trans_action(action)
        print()
        rent.move_car(str1, str2, num, verbose=True)
    return rent.get_Cash()


def read_from_file(filename, num_lines, start=0):
    data = []
    with open(filename, "r") as file:
        [next(file) for _ in range(start-1)]
        for idx in range(num_lines):
            sub_data = []
            str_line = file.readline()
            str_line = str_line.strip("]\n")
            for str_char in str_line.split(" "):
                try:
                    sub_data.append(int(str_char))
                except ValueError:
                    pass
            data.append(sub_data)
    return data

def plot_map_data(data):
    plt.figure()
    cs = plt.contourf(data, levels=list(range(-5, 6)), extend='both')
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()
    plt.xlabel("Store 1 num of cars")
    plt.ylabel("Store 2 num of cars")
    plt.show()


if __name__ == "__main__":
    rw = [[], []]
    policy = read_from_file("policy_list.txt", 21, 106)
    for _ in range(1000):
        # plot_map_data(policy)
        rw[0].append(run_car_rent(policy))
    for _ in range(1000):
        policy = [[np.random.randint(-5, 6, dtype=int) for _ in range(21)] for _ in range(21)]#
        # plot_map_data(policy)
        rw[1].append(run_car_rent(policy))
    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(rw[0])
    plt.subplot(1,2,2)
    plt.hist(rw[1])
    plt.show()
