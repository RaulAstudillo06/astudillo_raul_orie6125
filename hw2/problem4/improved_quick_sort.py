import numpy as np
from insertion_sort import insertion_sort


def improved_quick_sort(numbers):
    """
    Implements an improved version of quick sort, in which insertion sort is used whenever the size of the array is less than 8.
    :param numbers: [float] List of numbers to be sorted.
    :return: ([float]) Sorted list of the numbers.
    """

    if len(numbers) < 8:
        return insertion_sort(numbers)

    pivot = get_pivot(numbers)

    numbers, p = partition(numbers, pivot)
    number_pivot = numbers[p]

    numbers_l = improved_quick_sort(numbers[0: p])
    numbers_r = improved_quick_sort(numbers[p + 1:])
    return numbers_l + [number_pivot] + numbers_r


def get_pivot(numbers):
    """ 
    Returns the index of the median of the first, medium and last elements. This is used as pivot for quick sort.

    :param numbers:
    :return: int
    """
    r = len(numbers)-1
    m = int(np.floor(r/2))
    val_pivot = np.median([numbers[0], numbers[m], numbers[r]])
        
    for i in [0,m,r]:
        if numbers[i] == val_pivot:
            return i


def partition(numbers, pivot):
    """
    Partition step of quick sort.

    :param numbers:
    :param pivot: int
    :return: modified version of numbers and the split point of quick sort.
    """

    numbers = swap_numbers(numbers, 0, pivot)
    lo = 0
    hi = len(numbers) - 1

    while lo < hi:
        if numbers[lo + 1] <= numbers[lo]:
            numbers = swap_numbers(numbers, lo, lo + 1)
            lo += 1
        else:
            numbers = swap_numbers(numbers, lo + 1, hi)
            hi -= 1
    return numbers, lo


def swap_numbers(numbers, i, j):
    """
    Swap numbers[i] and numbers[j]

    :param numbers: [float]
    :param i: int
    :param j: int
    :return: [float]
    """

    assert i < len(numbers) and j < len(numbers)
        
    tmp = numbers[i]
    numbers[i] = numbers[j]
    numbers[j] = tmp
    return numbers