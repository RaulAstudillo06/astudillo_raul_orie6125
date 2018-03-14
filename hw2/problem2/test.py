from find_max_balanced_subarray import find_max_balanced_subarray


arr = ['a', 'a', 'z', 'z', 'z']
assert find_max_balanced_subarray(arr) == [0,3]

arr = ['a', 'a', 'a']
assert find_max_balanced_subarray(arr) == []

print('All tests were completed successfully.')
