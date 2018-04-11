import itertools
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)

# Number of friendship recommendations for each user
top_n_recommendations = 2
#
raw_social_net = sc.textFile('soc-data.txt')
social_net = raw_social_net.map(lambda line: line.replace('\t', ',')).map(lambda line: line.split(',')).map(lambda line: (line[0], line[1:]))

def createFriendsConnections( social_net_node):
  person = social_net_node[0]
  friends_list = social_net_node[1]
## Create existing connections    
  all_connections = [ ( ( each_friend, person ), 0) if person > each_friend else ( ( person, each_friend ), 0) for each_friend in friends_list ]
## Create possible future connections
  for friends_pair in itertools.combinations( friends_list, 2 ):
      if friends_pair[0] > friends_pair[1]:
          all_connections.append( ( ( friends_pair[1], friends_pair[0] ), 1 ) )
      else:
          all_connections.append( ( friends_pair, 1 ) )
  return all_connections

connections = social_net.flatMap(lambda node: createFriendsConnections(node))
num_common_friends = connections.groupByKey().filter(lambda item: 0 not in item[1]).map(lambda item: (item[0], sum(item[1]))).sortBy(lambda item: item[1], ascending=False)
individual_recommendations = num_common_friends.flatMap(lambda item: [(item[0][0], (item[0][1], item[1])), (item[0][1], (item[0][0], item[1]))] )
final_recommendations = individual_recommendations.groupByKey().mapValues(lambda item: sorted(item, key=lambda x: x[1], reverse=True)[:top_n_recommendations])
