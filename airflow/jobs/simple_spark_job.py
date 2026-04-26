# jobs/simple_spark_job.py
from pyspark.sql import SparkSession


spark = SparkSession.builder.appName('learning').getOrCreate()

text = "Hello, Spark! This is a simple Spark job. Hello Rama Hello Docker"

words = spark.sparkContext.parallelize(text.split(" "))

wordsCounts = words.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

for wc in wordsCounts.collect():
    print(f"{wc[0]}: {wc[1]}")


spark.stop()


# if __name__ == "__main__":
#     spark = (
#         SparkSession.builder.appName("SimpleSparkJob")
#         .getOrCreate()
#     )
#
#     data = [("Alice", 1), ("Bob", 2), ("Charlie", 3)]
#     columns = ["name", "id"]
#
#     df = spark.createDataFrame(data, columns)
#
#     print(f"Created a DataFrame with {df.count()} rows.")
#     df.show()
#
#     spark.stop()