def linear_model(x, w):
    return (w[2]*x[1])+(w[1]*x[0])+w[0]

def error(y, x, w, model):
    return y - model(x,w)

def loss(dataset, model, w):
    ret = 0.0
    for dato in dataset:
        e = error(dato[0], dato[1:], w, model)
        ret += e*e
    return ret

def gradient(fun, x, dataset, model, xi, delta=0.0001):   
    a = fun(dataset, model, x)
    xp = x
    xp[xi] += delta
    b = fun(dataset, model, xp)
    return (b-a)/delta

dataset = [
        [11, 4, 2],
        [7, 2, 2],
        [2, -1, 3],
        [1, 0, 0],
        [-2, -1, -1],
        ]

num_params = 3
params = [10.0] * num_params
l = 0.00001
avance = 1
params2 = params
while avance>0.000000000001:
   avance = 0
   for i, w in enumerate(params):
       wprev = w
       params2[i] = w - l*gradient(loss, params, dataset, linear_model, i)
       avance += abs(params2[i] - wprev)
   params = params2
   avance /= num_params

print(params)
