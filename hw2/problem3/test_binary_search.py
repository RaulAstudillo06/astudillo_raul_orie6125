from binary_search import binary_search

arr = [0,1,2,3,4,5,6,8]
item = 9
assert binary_search(arr, item) == None

arr = [0,1,2,3,4,5,6,8]
item = 4
assert binary_search(arr, item) == 4

print('All tests were completed successfully.')
