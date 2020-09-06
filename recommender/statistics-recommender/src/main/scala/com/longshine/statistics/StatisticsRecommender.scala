package com.longshine.statistics

import java.text.SimpleDateFormat
import java.util.Date

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}

// 1.样例类（推荐热门商品，只需要读取评分即可，商品信息不重要）
case class Rating(userID: Int, productId: Int, score: Double, timestamp: Long)

// 2.因为要保存到mogodb（这里不用mysql了），定义mogodb配置
case class MogoConfig(uri: String, db: String)

object StatisticsRecommender {
  // 3.定义mongodb中读取的表名
  val MONGODB_RATING_COLLECTION = "rating"
  // 4.分析的结果需要写入mongodb中
  // 评分更多的商品（历史热门统计）
  val RATE_MORE_PRODUCTS = "RateMoreProducts"
  // 最近评分更多的商品（近期热门统计）
  val RATE_MORE_RECENTLY_PRODUCTS = "RateMoreRecentlyProducts"
  // 平均评分统计（优质商品统计）
  val AVERAGE_PRODUCTS = "AverageProducts"


  def main(args: Array[String]): Unit = {
    // 5.定义配置
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://192.168.0.111:27017/recommender",
      "mongo.db" -> "recommender"
    )
    // 6.创建spark config
    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("StatisticsRecommender")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    // 7.加载数据
    import spark.implicits._
    implicit val mongoConfig = MogoConfig(config("mongo.uri"), config("mongo.db"))

    val ratingDF = spark
      .read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Rating]
      .toDF()

    // 创建一张名叫ratings的临时视图 （创建临时表的原因是，ratings数据结构是dataframe类型）
    // 统计是对视图数据进行统计，而不是在从mongodb中查询
    ratingDF.createOrReplaceTempView("ratings")

    // TODO 不同的统计推荐结果
    // 1.历史热门商品，按评分个数统计, productId, count
    val rateMoreProductsDF = spark.sql("select productId,count(productId) as count from ratings group by productId order by count desc")
    storeDFInMongoDB(rateMoreProductsDF, RATE_MORE_PRODUCTS)
    // 2.近期热门商品，,把时间戳转换为yyyyMM格式进行评分个数统计
    // 创建一个日期格式化工具
    val format = new SimpleDateFormat("yyyyMM")
    // 注册UDF，将timestamp转为yyyyMM(int类型）
    spark.udf.register("changeDate", (x: Int) => format.format(new Date(x * 1000L)).toInt)
    // 将原始rating数据转换为想要的结构 productId，count，score，yearmonth
    val ratingOfYearMonthDF = spark.sql("select productId, score,changeDate(timestamp) as yearmonth from ratings")
    ratingOfYearMonthDF.createOrReplaceTempView("ratingOfMonth")
    val rateMoreRecentlyPorudctsDF = spark.sql("select productId, count(productId) as count, yearmonth from ratingOfMonth group by yearmonth, productId order by yearmonth desc, count desc")
    storeDFInMongoDB(rateMoreRecentlyPorudctsDF, RATE_MORE_RECENTLY_PRODUCTS)
    // 3.优质商品统计，按商品的平均评分
    val averageProductsDF = spark.sql("select productId, avg(score) as avg from ratings group by productId order by avg desc")
    storeDFInMongoDB(averageProductsDF, AVERAGE_PRODUCTS)
    spark.stop()
  }

  // 保存到mongodb
  def storeDFInMongoDB(df: DataFrame, collectionName: String)(implicit mongoConfig: MogoConfig) = {
    df.write
      .option("uri", mongoConfig.uri)
      .option("collection", collectionName)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()
  }
}
