from random import randint
from scipy import integrate #imported library from https://docs.scipy.org/doc/scipy/reference/index.html
from scipy.misc import comb
from scipy.special import perm
from scipy.optimize import minimize as fnmin

class Binomial():
    def __init__(self, trials, prob):
        assert prob >= 0 and prob <= 1, "Probability must be between 0 and 1"
        assert isinstance(trials, int), "Number of trials must be integer"
        assert trials > 0, "Number of trials must be positive"
        self.n = trials
        self.p = prob

    def __repr__(self):
        return 'Binomial({0}, {1})'.format(self.n, self.p)

    def __str__(self):
        return 'A binomial random variable of {0} independent trials with a fixed probability {1}'.format(self.n, self.p)

    def event(self):
        trial = randint(0,100) / 100
        if trial <= self.p:
            return True
        return False

    def compute_events(self):
        successes, failures = 0, 0
        for _ in range(self.n):
            if self.event():
                successes += 1
            else:
                failures += 1
        return [successes, failures]

    def prob(self, k):
        assert isinstance(k, int), "event must occur an integer number of times"
        return comb(self.n, k) * (self.p ** k) * ((1 - self.p) ** (self.n-k))

    @property
    def expected_value(self):
        return self.n * self.p

    @property
    def variance(self):
        return (self.n * self.p) * (1 - self.p)

class Coin(Binomial):
    def __init__(self, trials, prob = 0.5):
        Binomial.__init__(self, trials, prob)
        self.heads = 0
        self.tails = 0

    def __str__(self):
        return "A coin with probability {0} of landing heads".format(self.p)

    def flip_coin(self):
        if self.event():
             self.heads += 1
             print('heads')
        else:
            self.tails += 1
            print('tails')

    def flip_coin_n(self):
        outcomes = self.compute_events()
        self.heads, self.tails = self.heads + outcomes[0], self.tails + outcomes[1]

    @property
    def outcomes(self):
        return self.heads, self.tails

class Learner():
    def __init__(self, n):
        self.guess = 0.5
        self.trials = n
        self.p = randint(0, 100) / 100
        self.coin = Coin(100, self.p)
        self.prior = lambda p: 1

    def __repr__(self):
        return "A computer learning the p-value of a coin"

    def avg_prob(self, suc = 1, fail = 0, density = lambda p: 1):
        """Calculate the average probability of a Bernoulli(p) event. Function
           takes the number of events that occur and the density of those events
           as a lambda function of p."""
        assert isinstance(suc, int), "Event must occur an integer number of times"
        assert isinstance(fail, int), "Event must fail an integer number of times"

        n = suc + fail
        prob = lambda p: pow(p, suc) * pow(1 - p, fail) #don't worry about the order of outcomes, it ends up cancelling out
        value = integrate.quad(lambda p: prob(p) * density(p), 0, 1)
        return value[0]

    def cond_density(self, suc = 1, fail = 0, prior = lambda p: 1):
        """Returns the conditional density of p such that given a set of outcomes
           the probability of different p values is returned as the posterior
           density."""

        numerator = lambda p: prior(p) * pow(p, suc) * pow(1 - p, fail)
        denominator = self.avg_prob(suc, fail, prior)
        posterior = lambda p: numerator(p) / denominator
        return posterior

    def learn(self):
        for _ in range(self.trials):
            self.coin.flip_coin()
            suc, fail = self.coin.outcomes[0], self.coin.outcomes[1]
            self.prior = self.cond_density(suc, fail, self.prior)
            #self.check_density(self.prior)
            self.guess = self.find_fn_max(self.prior)
            print(self.guess)
        print(self.p)


    def check_guess(self):
        if self.guess == self.p:
            return True
        return False

    def find_fn_max(self, fn):
        answer = fnmin(lambda x: -fn(x), self.guess, bounds = ((0,1),)).x
        return answer[0]

    def check_density(self, fn):
        answer = integrate.quad(fn, 0, 1)[0]
        if answer < 0.999 or answer > 1.0001: #Because it can integrate to .99999999...
            raise Exception('Density should integrate to 1')
