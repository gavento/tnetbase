

def argmax(iter, f):
    """
    Return `argmax(i in iter) f(i)`
    """
    best_v = None
    best_i = None
    for i in iter:
        v = f(i)
        if best_v is None or v > best_v:
            best_v = v
            best_i = i
    return best_i

