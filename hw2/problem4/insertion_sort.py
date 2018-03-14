
def insertion_sort(numbers):
    """
    Sort the list of numbers using insertion sort.
    :param numbers: ([float]) List of numbers.
    :return: ([float]) Sorted list of the numbers.
    """
    n = len(numbers)
        
    if n < 2:
        return numbers
        
    i =1 # Iinitialize pivot.
        
    while i < n:
        tmp = numbers[i]
        j = i-1
        # Find the right  place for tmp in sorted sublist.
        while j >= 0 and numbers[j] > tmp:
            numbers[j+1] = numbers[j] 
            j = j-1
        numbers[j+1] = tmp
        #
        i = i+1

    return numbers