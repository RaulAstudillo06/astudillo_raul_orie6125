import numpy as np


def binary_search(arr, item):
    """
    Standard bynary search algorithm.
    
    :param arr: list of numbers sorted in increasing order.
    : param item: number to be searched.
    """
    l = 0
    r = len(arr) -1
    
    while  l <= r :
        m = int(np.floor((l+r)/2))
        if arr[m] ==item:
            return m
        else:
            if item < arr[m]:
                r = m-1
            else:
                l = m+1
            
    print('{}  is not in input array.'.format(item))
    return None 
