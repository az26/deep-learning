import numpy as np

#seed = 1
#np.random.seed(seed)
#XOR

unit_shape = 4
input_shape = 2
output_shape = 2
e = 0.2
ep = 30000

input_wt = np.random.rand(unit_shape, input_shape)
output_wt = np.random.rand(output_shape , unit_shape)
b1 = np.random.rand(unit_shape,1)
b2 = np.random.rand(output_shape,1)
delta_hidden = np.arange(4).reshape((4,1))

x = np.array([[0,0],[0,1],[1,0],[1,1]]).reshape(4,2,1)
T = np.array([[1,0],[0,0],[0,0],[0,1]]).reshape(4,2,1)

def sigm(x):
    b =700
    if x >b:
        return 1/(1 + np.exp(-b))
    elif x<-b:
        return 1/(1 + np.exp(b))
    else:
        return 1/(1 + np.exp(-x))

def sig(a):
    sigmoid = np.vectorize(sigm)
    return sigmoid(a)

def train():
    global input_wt,output_wt,b1,b2
    s = 0
    while s < ep:
        r = 0
        while r < 4:
            X = x[r]
            t = T[r]

            u = input_wt.dot(X) + b1
            z = sig(u)
            y = sig(output_wt.dot(z)+b2)
            delta_out = y-t
            k = 0
            j = 0
            while j < unit_shape:
                while k < output_shape:
                    delta_hidden[j] = delta_out[k]*(output_wt[k,j]*(sig(u[j])*(1-sig(u[j]))))
                    k += 1
                j+= 1

            dout_wt = delta_out.dot(z.T)
            output_wt += -e*dout_wt

            dhidden_wt = delta_hidden.dot(X.T)
            input_wt += -e*dhidden_wt

            b2 += -e*delta_out
            b1 += -e*delta_hidden

            r += 1

        s += 1

def test():
    s = 0
    r = 0
    while r < 4:
        X = x[r]
        t = T[r]
        u = input_wt.dot(X) + b1
        z = sig(u)
        y = sig(output_wt.dot(z)+b2)
        print(X.T," ",t.T,"   ",y.T)
        r+= 1

    print(input_wt)
    print(output_wt)

train()
test()
