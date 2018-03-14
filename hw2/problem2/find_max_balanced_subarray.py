def find_max_balanced_subarray(arr):
    """
    Finds maximal contiguous subarray with same number of 'a's and 'z's.

    :param arr: list filled with 'a's and 'z's only.
    :return: first and last indices of maximal subarray if any, and an empty array otherwise.
    """
    l  = 0
    r = len(arr)-1
    
    n_a = arr.count('a')
    n_z = len(arr) - n_a
    
    if n_a > n_z:
        n_max = n_a
        n_min = n_z
        x = 'a'
    else:
        n_max = n_z
        n_min = n_a
        x = 'z'
    
    while n_max  > n_min and r - l > 0:
        # At every iteration we keep the subarray closes to be balanced from the two arrays obtained from
        # droping each of the extreme elements.
        if arr[l]==x:
            l += 1
            n_max -= 1
        else:
            if arr[r]==x:
                n_max -= 1
            else:
                n_min -= 1
            r -= 1
    
    # If l=r, there is no maximal subarray.
    if r - l > 0:
        return [l, r]
    else:
        return []