import sys
from pyspark.sql import SparkSession, functions, types, Row
import string, re
import math

spark = SparkSession.builder.appName('wordcount').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

def main(in_directory, out_directory):
    data = spark.read.text(in_directory)
    wordbreak = r'[%s\s]+' % (re.escape(string.punctuation),)  # regex that matches spaces and/or punctuation

    data = data.withColumn('word', functions.explode(functions.split('value', wordbreak)))
    data.cache()

    #Normalize all of the strings to lower-case
    data = data.withColumn('word', functions.lower(data['word']))
    data = data.select('word')


    #Notice that there are likely empty strings being counted: remove them from the output.
    data = data.filter(data['word'] != '')
    #count
    data = data.groupBy('word').agg(functions.count(data['word']))
    #data.show()

    #Sort by decreasing count (i.e. frequent words first) and alphabetically if there's a tie.
    data = data.sort(functions.desc('count(word)'), functions.asc('word'))
    data.show()
    data.write.csv(out_directory, mode = "overwrite")

if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
