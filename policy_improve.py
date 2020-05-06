from scipy.stats import poisson

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