import re
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)

trading_file = sc.textFile('taqtrade20141201.txt')
first_lines_trading_file = sc.parallelize(trading_file.take(10))

def sumDigits(s):
    # Returns sum of the digits of the number s (s is a string contanining numeric chars only).
    s_as_list = list(s)
    output = 0
    if len(s_as_list) > 0: 
        for d in s_as_list:
            output += int(d)
    return output
# return alphabetic characters and sum of digits in each line
output = first_lines_trading_file.map(lambda line: [re.sub('[^a-zA-Z]+', '', line), re.sub("[^0-9]", "", line)]).map(lambda line: [line[0], sumDigits(line[1])])

