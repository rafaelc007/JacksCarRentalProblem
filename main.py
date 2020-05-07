import numpy as np
from jackRental import JackRental, N_ACTIONS


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
    rent = JackRental([[3, 3, 5], [4, 2, 5]])
    for idx in range(100):
        rent.step()
        print("rent state:")
        print(rent)
        state = rent.get_state()
        action = policy[state[0]][state[1]]
        str1, str2, num = trans_action(action)
        print("moving {} from {} to {}\n".format(str1, str2, num))
        rent.move_car(str1, str2, num)


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

if __name__ == "__main__":
    policy = read_from_file("policy_list.txt", 21, 106)
    run_car_rent(policy)
