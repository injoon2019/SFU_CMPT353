import sys
from pyspark.sql import SparkSession, functions, types
import re

spark = SparkSession.builder.appName('wikipedia popular').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

wiki_schema = types.StructType([
    types.StructField('lang', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('views', types.LongType()),
    types.StructField('bytes', types.LongType()),
])

def date_search(path):
    date = re.search(r'\d{8}-\d{2}', path)
    return date.group(0)

def main(in_directory, out_directory):
	data = spark.read.csv(in_directory, schema=wiki_schema, sep=' ').withColumn('filename', functions.input_file_name())
	#data.show()

	data = data.filter(data['lang'] == 'en')
	data = data.filter(data['title'] != 'Main_Page')
	data = data.filter(data.title.startswith('Special:') != True)

	path_to_hour = functions.udf(lambda path: date_search(path), returnType=types.StringType())
	data = data.withColumn('time', path_to_hour(data.filename))

    #Use cache
	data = data.cache()

	groups = data.groupBy('time')
	max_views = groups.agg(functions.max(data['views']).alias('views'))

	data_join = max_views.join(data, on=['views','time'])
	output = data_join.drop('lang', 'bytes', 'filename')
	output = output.select('time', 'title', 'views')

	output = output.sort('time','title')
	output.show()

	output.write.csv(out_directory + '-wikipedia', mode='overwrite')
    #averages_by_subreddit.write.csv(out_directory + '-subreddit', mode='overwrite')
    #averages_by_score.write.csv(out_directory + '-score', mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
