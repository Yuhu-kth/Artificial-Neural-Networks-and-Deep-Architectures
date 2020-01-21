import numpy as np 

#implementation of Delta rule
def Delta_rule(X, T, W){
    """ epoch:
        eta:
        delta_W:
    """
    epoch = 20
    eta = 0.5
    delta_W = -eta*(W*X-T)*X.transpose

}