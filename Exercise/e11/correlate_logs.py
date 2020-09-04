import sys
from pyspark.sql import SparkSession, functions, types, Row
import re
import math

spark = SparkSession.builder.appName('correlate logs').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

line_re = re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred. Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        # TODO
        return Row(hostname=m.group(1), bytes=float(m.group(2)))
    else:
        return None


def not_none(row):
    """
    Is this None? Hint: .filter() with it.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    # TODO: return an RDD of Row() objects
    log_lines = log_lines.map(line_to_row)
    log_lines = log_lines.filter(not_none)
    return log_lines

def main(in_directory):
    logs = spark.createDataFrame(create_row_rdd(in_directory))

    # TODO: calculate r.
    # x is the number of requests made by that host, and y is the number of bytes transferred in those requests.
    num_counts = logs.groupby('hostname').count().cache()
    num_bytes = logs.groupby('hostname').sum('bytes').cache()
    calc_logs = num_counts.join(num_bytes, 'hostname')

    calc_logs = calc_logs.withColumnRenamed('count', 'Xi')
    calc_logs = calc_logs.withColumnRenamed('sum(bytes)', 'Yi')
    calc_logs = calc_logs.withColumn('Xi2', calc_logs['Xi']**2)
    calc_logs = calc_logs.withColumn('yi2', calc_logs['Yi']**2)
    calc_logs = calc_logs.withColumn('XiYi', calc_logs['Xi']*calc_logs['Yi'])


    result = calc_logs.groupby().sum()
    n = calc_logs.count()

    result = result.first()

    Xi = result[0]
    Yi = result[1]
    Xi2 = result[2]
    Yi2 = result[3]
    XiYi = result[4]

    r = (n*XiYi - Xi*Yi)/(math.sqrt(n*Xi2 - Xi**2)*math.sqrt(n*Yi2 - Yi**2))
    print("r = %g\nr^2 = %g" % (r, r**2))


if __name__=='__main__':
    in_directory = sys.argv[1]
    main(in_directory)
