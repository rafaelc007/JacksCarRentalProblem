from scipy.stats import poisson
import numpy as np
import sys


class poisson_:

    def __init__(self, lamb_param):
        self.lamb_param = lamb_param

        eps = 0.01

        # [alpha_param , beta_param] is the range of n's for which the pmf value is above eps
        self.alpha_param = 0
        state = 1
        self.vals = {}
        summer = 0

        while True:
            if state == 1:
                temp = poisson.pmf(self.alpha_param, self.lamb_param)
                if temp <= eps:
                    self.alpha_param += 1
                else:
                    self.vals[self.alpha_param] = temp
                    summer += temp
                    self.beta_param = self.alpha_param + 1
                    state = 2
            elif state == 2:
                temp = poisson.pmf(self.beta_param, self.lamb_param)
                if temp > eps:
                    self.vals[self.beta_param] = temp
                    summer += temp
                    self.beta_param += 1
                else:
                    break

                    # normalizing the pmf, values of n outside of [alpha_param, beta_param] have pmf = 0

        added_val = (1 - summer) / (self.beta_param - self.alpha_param)
        for key in self.vals:
            self.vals[key] += added_val

    def f(self, n):
        try:
            Ret_value = self.vals[n]
        except(KeyError):
            Ret_value = 0
        finally:
            return Ret_value


class location:
    # A class holding the properties of a location together
    def __init__(self, req, ret):
        self.req_mean = req  # value of lambda for requests
        self.ret_mean = ret  # value of lambda for returns
        self.poisson_req = poisson_(self.req_mean)
        self.poisson_ret = poisson_(self.ret_mean)


# Location initialisation
class ProblemDef:

    @staticmethod
    def max_cars():
        return 20

    @staticmethod
    def disc_fact():
        return 0.9

    @staticmethod
    def credit_reward():
        return 10

    @staticmethod
    def moving_reward():
        return -2

    @staticmethod
    def max_moving_cars():
        return 5
    
    A = location(3, 3)
    B = location(4, 2)

    value = np.zeros((max_cars()+1, max_cars()+1))
    policy = value.copy().astype(int)

    def expected_reward(self, state, action):
        """
        state  : It's a pair of integers, # of cars at A and at B
        action : # of cars transferred from A to B,  -5 <= action <= 5 
        """

        rw = 0  # reward
        new_state = [max(min(state[0] - action, self.max_cars()), 0), max(min(state[1] + action, self.max_cars()), 0)]

        # adding reward for moving cars from one location to another (which is negative)

        rw = rw + self.moving_reward() * abs(action)

        # there are four discrete random variables which determine the probability distribution of the reward and next state

        for Aa in range(self.A.poisson_req.alpha_param, self.A.poisson_req.beta_param):
            for Ba in range(self.B.poisson_req.alpha_param, self.B.poisson_req.beta_param):
                for Ab in range(self.A.poisson_ret.alpha_param, self.A.poisson_ret.beta_param):
                    for Bb in range(self.B.poisson_ret.alpha_param, self.B.poisson_ret.beta_param):
                        """
                        Aa : sample of cars requested at location A
                        Ab : sample of cars returned at location A
                        Ba : sample of cars requested at location B
                        Bb : sample of cars returned at location B
                        Prob  : probability of this event happening
                        """

                        # all four variables are independent of each other
                        Prob = self.A.poisson_req.vals[Aa] * self.B.poisson_req.vals[Ba] * \
                               self.A.poisson_ret.vals[Ab] * self.B.poisson_ret.vals[Bb]

                        valid_requests_A = min(new_state[0], Aa)
                        valid_requests_B = min(new_state[1], Ba)

                        rew = (valid_requests_A + valid_requests_B) * (self.credit_reward())

                        # calculating the new state based on the values of the four random variables
                        new_s = [0, 0]
                        new_s[0] = max(min(new_state[0] - valid_requests_A + Ab, self.max_cars()), 0)
                        new_s[1] = max(min(new_state[1] - valid_requests_B + Bb, self.max_cars()), 0)

                        # Bellman's equation
                        rw += Prob * (rew + self.disc_fact() * self.value[new_s[0]][new_s[1]])

        return rw

    # initial value of eps_param
    _eps_param = 50

    def policy_evaluation(self):

        # here policy_evaluation has a static variable eps_param whose values decreases over time
        eps_param = self._eps_param

        self._eps_param /= 10

        while True:
            delta_param = 0

            for i in range(self.value.shape[0]):
                for j in range(self.value.shape[1]):
                    # value[i][j] denotes the value of the state [i,j]

                    old_val = self.value[i][j]
                    self.value[i][j] = self.expected_reward([i, j], self.policy[i][j])

                    delta_param = max(delta_param, abs(self.value[i][j] - old_val))

                    print('.', end='')
                    sys.stdout.flush()
            print(delta_param)
            sys.stdout.flush()

            if delta_param < eps_param:
                break

    def policy_improvement(self):
        policy_stable = True
        for i in range(self.value.shape[0]):
            for j in range(self.value.shape[1]):
                old_action = self.policy[i][j]

                max_act_val = None
                max_act = None

                tal_12 = min(i, self.max_moving_cars())  # def boundaries
                tal_21 = -min(j, self.max_moving_cars())

                for act in range(tal_21, tal_12 + 1):
                    ksi_param = self.expected_reward([i, j], act)
                    if max_act_val is None:
                        max_act_val = ksi_param
                        max_act = act
                    elif max_act_val < ksi_param:
                        max_act_val = ksi_param
                        max_act = act

                self.policy[i][j] = max_act

                if old_action != self.policy[i][j]:
                    policy_stable = False

        return policy_stable

    def save_value(self):
        with open("value_list", "a") as file:
            file.write(str(self.value))

    def save_policy(self):
        with open("policy_list", "a") as file:
            file.write(str(self.policy))

if __name__ == "__main__":
    prob = ProblemDef
    while True:
        prob.policy_evaluation()
        param = prob.policy_improvement()
        prob.save_value()
        prob.save_policy()
        if param == True:
            break