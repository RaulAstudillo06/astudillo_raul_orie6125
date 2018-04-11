from pyspark import SparkConf, SparkContext
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import PCA, StandardScaler
import io

conf = SparkConf()
sc = SparkContext(conf=conf)

# Process the data
raw_votes = sc.textFile('114_congress.csv')

def remove_header(itr_index, itr):
    return iter(list(itr)[1:]) if itr_index == 0 else itr

votes = raw_votes.mapPartitionsWithIndex(remove_header).cache()
votes = votes.map(lambda line: [str(i + 1) + ":" + x + " " for i, x in enumerate(line.split(',')[3:])])
votes = votes.zipWithIndex().map(lambda line: [str(line[1]) + ' '] + line[0])
votes = votes.map(lambda line: (1,line + ['\n'])).map(lambda line: (line[0],''.join(line[1]))).reduceByKey(lambda v1, v2: v1+v2)
votes = votes.collect()[0][1]

encoding = 'utf-8'
with io.open('new_votes.txt', 'w', encoding=encoding) as file:
    file.write(votes)

dataset = spark.read.format('libsvm').load('new_votes.txt')

# K-means model.
kmeans = KMeans().setK(2).setSeed(1)
model = kmeans.fit(dataset)

# PCA
standardizer = StandardScaler(withMean=True, withStd=True, inputCol='features', outputCol='features_2')
model = standardizer.fit(dataset)
dataset = model.transform(dataset)
pca = PCA(k=2, inputCol='features_2', outputCol='pcaFeatures')
model = pca.fit(dataset)
result = model.transform(dataset).select('pcaFeatures')

# Putting all together
predictions = model.transform(dataset)
# ... (Not finished on time)
