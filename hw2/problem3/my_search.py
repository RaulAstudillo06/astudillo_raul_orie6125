import numpy as np
from binary_search import binary_search

def my_search(arr, item):
    """
    Searches a number in an array that has the form described in problem 3.
    
    :param arr: list of numbers as described in problem 3.
    :param item: number to be searched.
    """
    l = 0
    r = len(arr) -1
    
    while  r - l > 2:
        #print(l)
        #print(r)
        m = int(np.floor((l+r)/2))
        if arr[m] >arr[r]:
            if arr[l] <= item and item <= arr[m-1]:
                return l+binary_search(arr[l:m],item)
            else:
                l = m
        else:
            if arr[m+1] <= item and item <= arr[r]:
                return m+1+binary_search(arr[m+1:r+1],item)
            else:
                r = m
    
    for k in range(l,r+1):
        if arr[k] == item:
            return k
        
    print('{}  is not in input array.'.format(item))
    return None
