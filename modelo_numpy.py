import numpy as np

class Model:
    def __init__(self, num_params, model, lam, min_step):
        self.num_params = num_params
        self.params = np.array([10.0] * (num_params+1))
        self.model = model
        self.lam = lam
        self.min_step = min_step

    def __normailize(self, x):
        xp = x
        for i, v in enumerate(x):
            xp[i] = (v-self.media[i])/self.var[i]
        return xp

    def __error(self, y, x, params):
        return y - self.model(x,params[1:], self.params[0])

    def __loss(self, result, dataset, params):
        ret = 0.0
        for i, dato in enumerate(dataset):
            e = self.__error(result[i], dato, params)
            ret += e*e
        return ret

    def __gradient(self, result, dataset, xi, delta=0.0001):   
        a = self.__loss(result, dataset, self.params)
        xp = self.params
        xp[xi] += delta
        b = self.__loss(result, dataset, xp)
        return (b-a)/delta

    def __media(self, dataset):
        mu = np.array([])
        for d in np.transpose(dataset):
            mu = np.append(mu,np.mean(d))
        return mu

    def __varianza(self, dataset):
        var = np.array([])
        for d in np.transpose(dataset):
            var = np.append(var, np.var(d))
        return var
    
    def __normailize_dataset(self, data):
        ret = data
        for i, dato in enumerate(data):
            ret[i] = self.__normailize(dato)
        return ret

    def train(self, result, dataset):
        self.media = self.__media(dataset)
        self.var = self.__varianza(dataset)
        norm_data = self.__normailize_dataset(dataset)
        avance = 1
        params2 = self.params
        while avance>self.min_step:
           avance = 0
           for i, w in enumerate(self.params):
               wprev = w
               params2[i] = w - self.lam*self.__gradient(result, norm_data, i)
               avance += abs(params2[i] - wprev)
           self.params = params2
           avance /= self.num_params

    def solve(self, x):
        return self.model(self.__normailize(x), self.params[1:], self.params[0])

#######################################################################

def linear_model(x, w, b):
    return np.dot(x,w) + b

y = np.array([11.0, 7.0, 2.0, 1.0,-2.0])
x = np.array([
        [4.0, 2.0],
        [2.0, 2.0],
        [-1.0, 3.0],
        [0.0, 0.0],
        [-1.0, -1.0],
        ])

ia = Model(2, linear_model, 0.0001, 0.0000001)

print(ia.params)

ia.train(y, x)

print(ia.params)

print("6*2+4+1=17")
iadice = ia.solve([6, 4])
print(iadice)
print("Error = ", abs(17-iadice))

