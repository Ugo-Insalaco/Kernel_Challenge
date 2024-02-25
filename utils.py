from math import *
def list_to_tri_index(k, n):
    # i < j
    i = n - 2 - int(sqrt(-8*k + 4*n*(n-1)-7)/2.0 - 0.5)
    j = k + i + 1 - n*(n-1)/2 + (n-i)*((n-i)-1)/2
    return i, j

def tri_to_list_index(i, j, n):
    # i < j
    k = round((n*(n-1)/2) - (n-i)*((n-i)-1)/2 + j - i - 1)
    return k