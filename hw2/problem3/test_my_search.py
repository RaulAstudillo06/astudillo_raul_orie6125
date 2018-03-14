from my_search import my_search


arr = [15, 16, 17, 18, 19, 20, 11,12, 13]
item = 13
assert my_search(arr,item) == 8

arr = [15, 16, 17, 18, 19, 20, 11,12, 13]
item = 1
assert my_search(arr,item) == None

print('All tests were completed successfully.')