def linear_model(x, w):
    return w*x

def error(x, y, w, model):
    return y - model(x,w)

def loss(dataset, model, w):
    ret = 0.0
    for dato in dataset:
        e = error(dato[0], dato[1], w, model)
        ret += e*e
    return ret

def gradient(fun, x, dataset, model, delta=0.0001):   
    a = fun(dataset, model, x)
    b = fun(dataset, model, x+delta)
    return (b-a)/delta

dataset = [
        (0.0,0.0),
        (2.0,3.0),
        (-1.0,-1.5),
        ]

w = 10
l = 0.01
avance = 1
while avance>0.00000000001:
   wprev = w
   w = w - l*gradient(loss, w, dataset, linear_model)
   avance = abs(w - wprev)

print(w)
