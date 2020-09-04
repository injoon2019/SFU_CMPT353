import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit relative scores').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

comments_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('controversiality', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('id', types.StringType()),
    types.StructField('link_id', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('parent_id', types.StringType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('score_hidden', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('ups', types.LongType()),
    #types.StructField('year', types.IntegerType()),
    #types.StructField('month', types.IntegerType()),
])


def main(in_directory, out_directory):
    comments = spark.read.json(in_directory, schema=comments_schema)
    comments = comments.cache()
    # TODO
    groups = comments.groupBy('subreddit')
    averages = groups.agg(functions.avg(comments['score']).alias("average_score"))

    #Exclude any subreddits with average score ≤0.
    averages= averages.filter(averages.average_score >0)
    averages = averages.cache()

    #Join the average score to the collection of all comments. Divide to get the relative score.
    averages = comments.join(functions.broadcast(averages), 'subreddit')

    #We'll define “best” here as the comment with highest score relative to the average. That is, if a subreddit average score is 2.3 and a particular comment has score 4, then the relative score is 4/2.3 = 1.74.
    averages = averages.withColumn('relative_score', averages['score']/averages['average_score'])

    #Determine the max relative score for each subreddit.
    averages = averages.groupBy('subreddit').agg(functions.max(averages.relative_score).alias("rel_score"))

    #Join again to get the best comment on each subreddit: we need this step to get the author.
    best_author = groups.agg(functions.max(comments['score']).alias("score"))
    best_author = best_author.cache()

    # Join again to get the best comment on each subreddit: we need this step to get the author.
    best_author = comments.join(functions.broadcast(best_author), ['score','subreddit'])

    best_author = best_author.join(averages, 'subreddit')
    best_author = best_author.select('subreddit', 'author', 'rel_score')

    best_author.write.json(out_directory, mode='overwrite')


    best_author.write.json(out_directory, mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)
