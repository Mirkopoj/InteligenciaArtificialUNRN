import numpy as np

class Model:
    def __init__(self, num_params, model, lam, min_step):
        self.num_params = num_params
        self.params = np.array([10.0] * num_params)
        self.model = model
        self.lam = lam
        self.min_step = min_step

    def __normailize(self, x):
        xp = x
        for i, v in enumerate(x):
            xp[i] = (v*self.media[i])/self.var[i]
        return xp

    def __error(self, y, x, params):
        return y - self.model(self.__normailize(x),params)

    def __loss(self, result, dataset, params):
        ret = 0.0
        for dato in dataset:
            e = self.__error(result, dato,params)
            ret += e*e
        return ret

    def __gradient(self, result, dataset, xi, delta=0.0001):   
        a = self.__loss(result, dataset, self.params)
        xp = self.params
        xp[xi] += delta
        b = self.__loss(result, dataset, xp)
        return (b-a)/delta

    def __media(self, dataset):
        mu = [0.0] * self.num_params
        for d in dataset:
            for i, e in enumerate(d):
                mu[i] += e
        for i in range(len(dataset[0])):
            mu[i] /= len(dataset)
        return mu

    def __varianza(self, dataset):
        var = [0.0] * self.num_params
        for d in dataset:
            for i, e in enumerate(d):
                v = e-self.media[i]
                var[i] += v*v
        for i in range(len(dataset[0])):
            var[i] /= len(dataset)
        return var

    def train(self, result, dataset):
        self.media = self.__media(dataset)
        self.var = self.__varianza(dataset)
        avance = 1
        params2 = self.params
        while avance>self.min_step:
           avance = 0
           for i, w in enumerate(self.params):
               wprev = w
               params2[i] = w - self.lam*self.__gradient(result, dataset, i)
               avance += abs(params2[i] - wprev)
           self.params = params2
           avance /= self.num_params

    def solve(self, x):
        return self.model(self.__normailize(x), self.params)

#######################################################################

def linear_model(x, w):
    return (w[2]*x[1])+(w[1]*x[0])+w[0]

y = np.array([11, 7, 2, 1,-2])
x = np.array([
        [4, 2],
        [2, 2],
        [-1, 3],
        [0, 0],
        [-1, -1],
        ])

ia = Model(3, linear_model, 0.0001, 0.0000001)

print(ia.params)

ia.train(y, x)

print(ia.params)

print("6*2+4+1=17")
iadice = ia.solve([6, 4])
print(iadice)
print("Error = ", abs(17-iadice))

