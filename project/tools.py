########### Useful functions #################################################################

#First note that we in fact computer the entropy of y at question 1 and it is better to avoid to recompute it at each call
#It may be better to compute it on the complete dataset but train is good enough and faster 
def getHy(y_train):
    y_data = np.genfromtxt(y_train, delimiter=',', skip_header=1)
    y_mean = y_data[:,0].mean()
    Hy = -y_mean * log( y_mean) - (1-y_mean) *log(1-y_mean)
    return Hy


# Dot function
# INPUT:
#     A: first vector
#     B: second vector
# OUTPUT:
#     product <A,B>
def dot(A,B):
    if len(A) != len(B):
      return 0
    else:
      sum = 0
      for i in range(len(A)):
        sum += A[i]*B[i]
      return sum

# Logloss
# INPUT:
#     p: our prediction
#     y: real answer
# OUTPUT
#     logarithmic loss of p given y
def logloss(p, y):
    p = max(min(p, 1. - 10e-12), 10e-12)
    return -log(p) if y == 1. else -log(1. - p)

def exec_time(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(method.__name__, ': execution time:',(te - ts))
        return result
    return timed