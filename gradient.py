def gradient(fun, x, delta=0.0001):   
    a = fun(x)
    b = fun(x+delta)
    return (b-a)/delta

def f(x):
    return x*x + 3*x -1
    
print(gradient(f, 4))
