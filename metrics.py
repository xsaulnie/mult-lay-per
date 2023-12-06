import numpy as np

def statmetrics(Y, Y_hat):
    if (Y.shape[0] != Y_hat.shape[0]):
        return None
    st = {'tp' : 0, 'tn' : 0, 'fp' : 0, 'fn' :0}
    for idx in range(Y.shape[0]):
        if (Y_hat[idx] == Y[idx]):
            if Y_hat[idx] == 1:
                st['tp'] = st['tp'] + 1
            else:
                st['tn'] = st['tn'] + 1
        else:
            if Y[idx] == 1:
                st['fn'] = st['fn'] + 1
            else:
                st['fp'] = st['fp'] + 1
    return (st)

def precision(Y, Y_hat):
    if (Y_hat.shape[0] != Y.shape[0]):
        return None
    st = statmetrics(Y, Y_hat)
    if (st['tp'] == 0):
        return float(0)
    return st['tp'] / (st['tp'] + st['fp'])

def recall(Y, Y_hat):
    if (Y_hat.shape[0] != Y.shape[0]):
        return None
    st = statmetrics(Y, Y_hat)
    if (st['tp'] == 0):
        return float(0)
    return st['tp'] / (st['tp'] + st['fn'])

def f1_score(Y, Y_hat):
    if (Y_hat.shape[0] != Y.shape[0]):
        return None
    prec = precision(Y, Y_hat)
    reca = recall(Y, Y_hat)
    if (prec == 0 or reca == 0):
        return float(0)
    if (prec + reca == 0):
        return float('inf')

    return((2 * prec * reca) / (prec + reca))

def getmetrics(Y, Y_hat):
    if (Y.shape[0] != Y_hat.shape[0]):
        return None
    SY = np.array([1 if i[0] > i[1] else 0 for i in Y])
    SY_hat = np.array([1 if i[0] > i[1] else 0 for i in Y_hat])

    return(precision(SY, SY_hat), recall(SY, SY_hat), f1_score(SY, SY_hat))


            
