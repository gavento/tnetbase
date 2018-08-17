#!/usr/bin/env python3


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


def index(iterable):
    idx = {}
    for i, it in enumerate(iterable):
        idx[it] = i
    return idx
