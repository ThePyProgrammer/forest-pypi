import scipy.optimize as op
import np

def sigmoid(z):
    return 1/(1 + np.exp(-z));


def linReg(X, y, theta, λ=0):
    m = y.T.shape
    h = (X*theta)-y
    regTheta = theta[:,:]
    regTheta[1] = 0
    J = (dot(h, h) + λ*dot(regTheta, regTheta))/(2*m)
    grad = (X.T*h + λ*regTheta)/m
    return J, grad

def logReg(theta, X, Y, Lambda):
    m = len(Y)
    y = Y[:,np.newaxis]
    h = sigmoid(X @ theta)
    cost = sum((-y * np.log(h)) - ((1-y)*np.log(1-h)))/m
    regCost= cost + Lambda/(2*m) * (theta @ theta - theta[0]**2)
    
    # compute gradient
    j_0 = 1/m * (X.transpose() @ (h - y))[0]
    j_1 = ((X.transpose() @ (h - y))[1:] + Lambda* theta[1:])/m
    grad= np.vstack((j_0[:,np.newaxis],j_1))
    return regCost[0], grad

def fminunc(f, X, y, initial_theta=None):
    return op.fmin_tnc(func=lambda theta, X, y: tuple(f(theta)), x0 = np.zeros(X.shape[1]) if initial_theta is None else initial_theta, args = (X, y))[0]

def gradientDescent(func, theta, alpha=0.01, num_iters=10):
    J_history =[]
    for i in range(num_iters):
        cost, grad = func(theta)
        theta = theta - (alpha * grad)
        J_history.append(cost)
    return theta, J_history

import numpy.linalg as la
def div(a, b):
    return la.solve(b.T, a.T).T

class optimset:
    def __init__(self, **kwargs):
        self.diction = kwargs

    def __getitem__(self, key): return self.diction[key]
    def __setitem__(self, key, value): self.diction[key] = value
    
    

def fmincg(f, X, options=optimset(MaxIter=200, GradObj='on')):

    if options['MaxIter']:
        length = options['MaxIter']
    else:
        length = 100    
    
    RHO = 0.01                            # a bunch of constants for line searches
    SIG = 0.5       # RHO and SIG are the constants in the Wolfe-Powell conditions
    INT = 0.1    # don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0                    # extrapolate maximum 3 times the current bracket
    MAX = 20                         # max 20 function evaluations per line search
    RATIO = 100                                      # maximum allowed slope ratio

    # FIXME
    red = 1

    i = 0                                            # zero the run length counter
    ls_failed = 0                             # no previous line search has failed
    fX = np.array([])
    f1, df1 = f(X)                      # get function value and gradient
    i = i + (length<0)                                            # count epochs?!
    s = -df1                                        # search direction is steepest
    d1 = np.dot(-s.T, s)                                                 # this is the slope
    z1 = red/(1-d1)                                  # initial step is red/(|s|+1)
    
    while i < abs(length):                           # while not finished
        i = i + (length>0)                           # count iterations?!
        
        X0 = X; f0 = f1; df0 = df1                   # make a copy of current values                                            # begin line search
        X = X + np.multiply(z1,s)                   # begin line search    
        f2, df2 = f(X)
        i = i + (length<0)                                          # count epochs?!
        d2 = np.dot(df2.T, s)
        f3 = f1
        d3 = d1
        z3 = -z1             # initialize point 3 equal to point 1
        if length>0:
            M = MAX
        else:
            M = min(MAX, -length-i)
        success = 0
        limit = -1                     # initialize quanteties
        while True:
            while ((f2 > f1+np.dot(np.dot(z1,RHO),d1)) or (d2 > np.dot(-SIG, d1))) and (M > 0):
                limit = z1                                         # tighten the bracket
                if f2 > f1:
                    z2 = z3 - (0.5*np.dot(np.dot(d3,z3),z3))/(np.dot(d3, z3)+f2-f3)                 # quadratic fit
                else:
                    A = 6*(f2-f3)/z3+3*(d2+d3)                                 # cubic fit
                    B = 3*(f3-f2)-np.dot(z3,(d3+2*d2))
                    z2 = (np.sqrt(np.dot(B, B)-np.dot(np.dot(np.dot(A,d2),z3),z3))-B)/A       # numerical error possible - ok!
                if isnan(z2) | isinf(z2):
                    z2 = z3/2                  # if we had a numerical problem then bisect
                z2 = max(min(z2, INT*z3),(1-INT)*z3)  # don't accept too close to limits
                z1 = z1 + z2                                           # update the step
                X = X + np.multiply(z2,s)
                f2, df2 = f(X)
                M = M - 1
                i = i + (length<0)                           # count epochs?!
                d2 = np.dot(np.transpose(df2),s)
                z3 = z3-z2                    # z3 is now relative to the location of z2    
            if (f2 > f1+np.dot(z1*RHO,d1)) or (d2 > -SIG*d1):
                break                                                # this is a failure
            elif d2 > SIG*d1:
                success = 1
                break                                             # success
            elif M == 0:
                break                                                          # failure
            A = 6*(f2-f3)/z3+3*(d2+d3)                      # make cubic extrapolation
            B = 3*(f3-f2)-np.dot(z3, (d3+2*d2))
            z2 = -np.dot(np.dot(d2,z3),z3)/(B+np.sqrt(np.dot(B,B)-np.dot(np.dot(np.dot(A,d2),z3),z3)))        # num. error possible - ok!
            if z2 is not float or isnan(z2) or isinf(z2) or z2 < 0:   # num prob or wrong sign?
                if limit < -0.5:                               # if we have no upper limit
                    z2 = z1 * (EXT-1)                 # the extrapolate the maximum amount
                else:
                    z2 = (limit-z1)/2                                   # otherwise bisect
            elif (limit > -0.5) and (z2+z1 > limit):          # extraplation beyond max?
                z2 = (limit-z1)/2                                               # bisect
            elif (limit < -0.5) and (z2+z1 > z1*EXT):       # extrapolation beyond limit
                z2 = z1*(EXT-1.0)                           # set to extrapolation limit
            elif z2 < -z3*INT:
                z2 = -z3*INT
            elif (limit > -0.5) and (z2 < (limit-z1)*(1.0-INT)):   # too close to limit?
                z2 = (limit-z1)*(1.0-INT)
            f3 = f2; d3 = d2; z3 = -z2                  # set point 3 equal to point 2
            z1 = z1 + z2
            X = X + np.multiply(z2,s)                      # update current estimates
            f2, df2 = f(X)
            M = M - 1
            i = i + (length<0)                             # count epochs?!
            d2 = np.dot(df2.T,s)
        if success:                                         # if line search succeeded
            f1 = f2
##            print (fX.T).shape
##            print isinstance(f1, np.generic)
            fX = np.append((fX.T, [float(f1)]) ,1).T
##            fX = np.concatenate(([fX.T], [f1]) ,1).T            
            print("Iteration %i | Cost: %f \r" %(i,f1)),            
            s = np.multiply((np.dot(df2.T,df2)-np.dot(df1.T,df2))/(np.dot(df1.T,df1)), s) - df2      # Polack-Ribiere direction
            tmp = df1
            df1 = df2
            df2 = tmp                         # swap derivatives
            d2 = np.dot(df1.T,s)
            if d2 > 0:                                      # new slope must be negative
                s = -df1                              # otherwise use steepest direction
                d2 = np.dot(-s.T,s)
            z1 = z1 * min(RATIO, d1/(d2-sys.float_info.min))          # slope ratio but max RATIO
            d1 = d2
            ls_failed = 0                              # this line search did not fail
        else:
            X = X0
            f1 = f0
            df1 = df0  # restore point from before failed line search
            if ls_failed or (i > abs(length)): # line search failed twice in a row
                break                             # or we ran out of time, so we give up
            tmp = df1
            df1 = df2
            df2 = tmp                         # swap derivatives
            s = -df1                                                    # try steepest
            d1 = np.dot(-s.T,s)
            z1 = 1/(1-d1)                     
            ls_failed = 1                                    # this line search failed
    print()
    return X, fX, i
