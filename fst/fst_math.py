

import math

kLogZero = -float('inf')
FLT_EPSILON = 1.19209290e-7
kMinLogDiffFloat = math.log(1.19209290e-7)

def Log1p(x):
    cutoff = 1.0e-08
    if x < cutoff:
        return x - 0.5 * x * x
    else:
        return Log(1.0 + x)


def LogAdd(x, y):
    if x < y:
        diff = x - y
        x = y
    else:
        diff = y - x
    # diff is negative.  x is now the larger one.

    if diff >= kMinLogDiffFloat:
        res = x + math.log1p(math.exp(diff))
        return res
    else:
        return x # return the larger one

