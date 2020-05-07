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

    _max_cars = 20
    _disc_fact = 0.9
    _credit_reward = 10
    _moving_reward = -2
    _max_moving_cars = 5
    
    A = location(3, 3)
    B = location(4, 2)

    value = np.zeros((_max_cars+1, _max_cars+1))
    policy = value.copy().astype(int)

    def set_values(self, val):
        self.value = np.array(val)

    def set_policy(self, pol):
        self.policy = np.array(pol)

    def expected_reward(self, state, action):
        """
        state  : It's a pair of integers, # of cars at A and at B
        action : # of cars transferred from A to B,  -5 <= action <= 5 
        """

        rw = 0  # reward
        new_state = [max(min(state[0] - action, self._max_cars), 0), max(min(state[1] + action, self._max_cars), 0)]

        # adding reward for moving cars from one location to another (which is negative)

        rw = rw + self._moving_reward * abs(action)

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

                        rew = (valid_requests_A + valid_requests_B) * (self._credit_reward)

                        # calculating the new state based on the values of the four random variables
                        new_s = [0, 0]
                        new_s[0] = max(min(new_state[0] - valid_requests_A + Ab, self._max_cars), 0)
                        new_s[1] = max(min(new_state[1] - valid_requests_B + Bb, self._max_cars), 0)

                        # Bellman's equation
                        rw += Prob * (rew + self._disc_fact * self.value[new_s[0]][new_s[1]])

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

                tal_12 = min(i, self._max_moving_cars)  # def boundaries
                tal_21 = -min(j, self._max_moving_cars)

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
        with open("value_list.txt", "a") as file:
            print("saving file...")
            file.write(str(self.value)+"\n")

    def save_policy(self):
        with open("policy_list.txt", "a") as file:
            print("saving file...")
            file.write(str(self.policy)+"\n")

if __name__ == "__main__":
    prob = ProblemDef()
    prob.set_policy([[ 0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -2, -3, -3, -3, -4, -4, -4, -4, -5, -5], \
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -2, -2, -2, -3, -3, -3, -3, -4, -4, -4], \
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -2, -2, -2, -2, -3, -3, -3, -3], \
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -2, -2, -2, -2, -3], \
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -2, -2], \
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -1, -1], \
 [ 1,  1,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1], \
 [ 2,  2,  2,  2,  2,  1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], \
 [ 3,  3,  3,  3,  2,  2,  2,  2,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], \
 [ 4,  4,  4,  3,  3,  3,  3,  2,  2,  2,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0], \
 [ 5,  5,  4,  4,  4,  4,  3,  3,  3,  2,  2,  2,  1,  1,  1,  0,  0,  0,  0,  0,  0], \
 [ 5,  5,  5,  5,  5,  4,  4,  4,  3,  3,  3,  2,  2,  2,  1,  1,  0,  0,  0,  0,  0], \
 [ 5,  5,  5,  5,  5,  5,  5,  4,  4,  4,  3,  3,  3,  2,  2,  1,  1,  0,  0,  0,  0], \
 [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  4,  4,  4,  3,  3,  2,  2,  1,  0,  0,  0,  0], \
 [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  4,  4,  3,  3,  2,  1,  0,  0,  0,  0], \
 [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  4,  4,  3,  2,  1,  0,  0,  0,  0], \
 [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  4,  3,  2,  1,  0,  0,  0,  0], \
 [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  4,  3,  2,  1,  0,  0,  0,  0], \
 [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  4,  3,  2,  1,  0,  0,  0,  0], \
 [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  4,  3,  2,  1,  0,  0,  0,  0], \
 [ 5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  4,  3,  2,  1,  1,  0,  0,  0]])
    prob.set_values([[217.21661395, 227.42986664, 237.80452556, 248.04779104, 257.98167995,  267.57178505, 276.80043011, 285.6349218 , 294.04822844, 302.02920057,  309.57597306, 316.69154493, 323.37928763, 329.64244408, 335.48323042,  340.90297838, 345.90076531, 350.46908195, 354.58669056, 358.21161553,  361.28280548], \
 [228.77515914, 239.38355886, 250.48397678, 261.32889705, 271.554496,  281.23343018, 290.47549772, 299.3057729 , 307.71353599, 315.69295427,  323.24345965, 330.36932194, 337.07485541, 343.36405768, 349.23964376,  354.70310833, 359.75326429, 364.38179408, 368.56595012, 372.26137964,  375.40375834], \
 [242.53031652, 253.37169637, 264.8976408 , 276.0901814 , 286.47781122,  296.19817691, 305.43751027, 314.25647259, 322.65540639, 330.63268895,  338.18932951, 345.33114279, 352.06347399, 358.39115649, 364.31744063,  369.84390437, 374.96879737, 379.68233296, 383.95904796, 387.75031064,  390.98602094], \
 [256.20054915, 267.02175188, 278.50886916, 289.66465407, 300.0265491,  309.7282271 , 318.95200141, 327.75834276, 336.14909263, 344.12414806,  351.6858715 , 358.84139659, 365.59693719, 371.95803531, 377.92839587,  383.5096499 , 388.69949271, 393.4866883 , 397.84305376, 401.71563806,  405.02855259], \
 [267.60375444, 278.33373682, 289.6522761 , 300.66518424, 310.95133118,  320.62060232, 329.82834706, 338.62422965, 347.00836433, 354.98097165,  362.54534206, 369.70952043, 376.48031958, 382.86378809, 388.86397776,  394.48259466, 399.71697557, 404.55486425, 408.96610048, 412.89454506,  416.26002034], \
 [276.57342147, 287.2496774 , 298.46854411, 309.39602425, 319.63535487,  329.28275817, 338.47819858, 347.26542888, 355.64383285, 363.61394489,  371.17963512, 378.34953364, 385.13080348, 391.52981844, 397.55087967,  403.19577616, 408.46163433, 413.33552149, 417.78591447, 421.75443343,  425.15785062], \
 [283.75233021, 294.41089896, 305.59630675, 316.49349553, 326.71362037,  336.34902582, 345.53555998, 354.31593589, 362.68993831, 370.65857421,  378.22611216, 385.40158887, 392.19237417, 398.60505145, 404.6440936,  410.31134658, 415.60377271, 420.50791205, 424.99117049, 428.99339387,  432.42891973], \
 [289.62056646, 300.27595462, 311.45476464, 322.34457646, 332.5579391,  342.18714322, 351.36824869, 360.1447041 , 368.5170479 , 376.48679632,  384.05855091, 391.24166818, 398.04366915, 404.47127763, 410.52906735,  416.21888547, 421.53750772, 426.47097648, 430.98572811, 435.02002193,  438.48602434], \
 [294.44761194, 305.10418154, 316.28475816, 327.17508582, 337.38730803,  347.01467386, 356.19447922, 364.97118769, 373.34604998, 381.32105313,  388.90110751, 396.09582827, 402.91286202, 409.35901314, 415.43887586,  421.15421778, 426.50156429, 431.4664256 , 436.01427239, 440.08182431,  443.57916241], \
 [298.43544383, 309.09475632, 320.28033993, 331.1748842 , 341.38967271,  351.01923535, 360.20230773, 368.9842935 , 377.36701731, 385.3528515,  392.94695698, 400.15910754, 406.99701204, 413.46746291, 419.57496692,  425.32110595, 430.70206305, 435.70274377, 440.28759979, 444.39177428,  447.92324071], \
 [301.77662877, 312.43992299, 323.63315864, 334.5351446 , 344.7560757,  354.39209044, 363.58354248, 372.37669937, 380.77381368, 388.77752659,  396.39314822, 403.63046808, 410.49715638, 416.99987575, 423.14292328,  428.92758013, 434.34958288, 439.39313251, 444.0215519 , 448.16827281,  451.73900086], \
 [304.64891046, 315.31706177, 326.51988346, 337.43186964, 347.66215737,  357.30888202, 366.51422378, 375.3252601 , 383.74449442, 391.77463183,  399.42092159, 406.69293435, 413.5981267 , 420.14287566, 426.33113847,  432.1637905 , 437.63603158, 442.7312621 , 447.41154667, 451.608417,  455.225063  ], \
 [307.20850821, 317.88197198, 329.09552508, 340.01941828, 350.26208648,  359.92409333, 369.14962324, 377.98645443, 386.43705221, 394.50381996,  402.1915757 , 409.5093163 , 416.4640353 , 423.06164914, 429.30566829,  435.19650472, 440.72878509, 445.88505934, 450.62603169, 454.88114641,  458.5508056 ], \
 [309.59342375, 320.2721725 , 331.4968749 , 342.43444082, 352.69337722,  362.37694843, 371.63116626, 380.50406685, 388.99751677, 397.11304499,  404.85453578, 412.22998005, 419.24567204, 425.90695238, 432.21686707,  438.17540917, 443.77669699, 449.00247026, 453.81204419, 458.13264339,  461.86164735], \
 [311.92354722, 322.6070479 , 333.84286276, 344.79675044, 355.07855299,  364.79395661, 374.08970272, 383.01286328, 391.56367834, 399.74204602,  407.55042516, 414.99542775, 422.08260685, 428.81680407, 435.20075825,  441.23424424, 446.91107495, 452.21234083, 457.09604559, 461.48715858,  465.27989042], \
 [314.29278307, 324.98007149, 336.22701106, 347.20208093, 357.51818356,  367.28209964, 376.63862005, 385.63136976, 394.25731897, 402.51392208,  410.40198867, 417.92665638, 425.09312791, 431.90616403, 438.36860227,  444.48038265, 450.23535862, 455.61425312, 460.57394851, 465.03723961,  468.89514402], \
 [316.75715613, 327.4470453 , 338.70587904, 349.71056636, 360.07908469,  369.91643531, 379.36060495, 388.44765195, 397.16899408, 405.51902164,  413.49728734, 421.10790692, 428.35680013, 435.24952807, 441.78970839,  447.97798596, 453.80869923, 459.26256453, 464.29561811, 468.82866923,  472.7496502 ], \
 [319.31907118, 330.01046324, 341.28334541, 352.33024572, 362.77617843,  372.71972163, 382.28489791, 391.49446372, 400.33156868, 408.78742846,  416.86165722, 424.55856051, 431.88663839, 438.85360797, 445.46476176,  451.72205007, 457.62074196, 463.14190611, 468.2410358 , 472.83722045,  476.8155377 ], \
 [321.90662282, 332.59879661, 343.88915916, 354.99324616, 365.54453312,  375.62934368, 385.34930573, 394.70744371, 403.6760515 , 412.24378075,  420.41252332, 428.18866766, 435.585712  , 442.6151339 , 449.28481641,  455.59851791, 461.55275725, 467.12921624, 472.28311107, 476.93211696,  480.95879834], \
 [324.36369552, 335.05640392, 346.36760679, 357.54118254, 368.21946567,  378.47121876, 388.3685434 , 397.88962983, 406.99433168, 415.66996032,  423.92317764, 431.76448489, 439.21461163, 446.29014297, 453.00218758,  459.35654452, 465.35107068, 470.96815556, 476.16292324, 480.85195635,  484.91575223], \
 [326.48400154, 338.81767577, 353.104494  , 366.68758069, 378.53136966,  389.16852753, 399.1592831 , 408.69046842, 417.77655273, 426.4217927,  434.63704929, 442.43709348, 449.84690933, 456.88650365, 463.569626,  469.90408679, 475.88918647, 481.50808606, 486.71574899, 491.42732525,  495.52039637]])
    while True:
        prob.policy_evaluation()
        param = prob.policy_improvement()
        prob.save_value()
        prob.save_policy()
        if param == True:
            break