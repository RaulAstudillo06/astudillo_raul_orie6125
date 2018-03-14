from bellman_ford import bellman_ford


n = 4
edges = [[0,1], [1,2], [2,3], [3,0]]
weights = [1, 1, 1, 1]
assert bellman_ford(n,edges,weights) == False

n = 4
edges = [[0,1], [1,2], [2,3], [3,0]]
weights = [1, 1, 1, -4]
assert bellman_ford(n,edges,weights) == True

print('All tests were completed successfully.')