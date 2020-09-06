package com.longshine

import breeze.numerics.sqrt
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

// 1.样例类
case class Product_Rating(userID: Int, productId: Int, score: Double, timestamp: Long)

case class MogoConfig(uri: String, db: String)

// 2.用于保存的样例类
// 定义推荐对象
case class Recommendation(productId: Int, score: Double)

// 定义用户的推荐列表
case class UserRecs(userId: Int, recs: Seq[Recommendation])

// 定义商品相似度列表
case class ProductRecs(productId: Int, recs: Seq[Recommendation])

object ALSTrainer {
  // 3.读取的表
  val MONGODB_RATING_COLLECTION = "rating"
  // 4.保存到表中的表名
  // 用户推荐列表
  val UserRecs_COLLECTION = "UserRecs"
  // 相似度列表
  val ProductRecs_COLLECTION = "ProductRecs"
  // 推荐个数
  val USER_MAX_RECOMMENDATION = 20


  def main(args: Array[String]): Unit = {
    // 4.定义配置
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://192.168.0.111:27017/recommender",
      "mongo.db" -> "recommender"
    )
    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("OfflineRecommender")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    // 5.加载数据
    import spark.implicits._
    implicit val mongoConfig = MogoConfig(config("mongo.uri"), config("mongo.db"))
    // als需要rdd
    val ratingRDD = spark
      .read
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .format("com.mongodb.spark.sql")
      .load()
      .as[Product_Rating]
      .rdd
      .map(
        rating => Rating(rating.userID, rating.productId, rating.score)
      ).cache()
    // 数据集切分成训练集和测试集
    val splits = ratingRDD.randomSplit(Array(0.8, 0.2))
    val trainingRDD = splits(0)
    val testingRDD = splits(1)

    // 核心实现，输出最优参数
    adjustALSParams(trainingRDD, testingRDD)
    spark.stop()
  }


  def adjustALSParams(trainData: RDD[Rating], testData: RDD[Rating]): Unit = {
    // 遍历数组中定义的参数取值
    val result = for (rank <- Array(5, 10, 20, 50); lambda <- Array(1, 0.1, 0.01))
      yield {
        val model = ALS.train(trainData, rank, 10, lambda)
        val rmse = getRMSE(model, testData)
        (rank, lambda, rmse)
      }
    // 按照rmse排序并输出最优参数
    // (5,0.1,1.2879352903612808)
    println(result.minBy(_._3))
  }

  def getRMSE(model: MatrixFactorizationModel, testData: RDD[Rating]): Double = {
    // 构建userProducts得到预测评分矩阵
    val userProducts = testData.map(item => (item.user, item.product))
    val predictRating = model.predict(userProducts)
    // 按照公式计算rmse,把预测评分和实际评分表按userId和productId做连接
    // 真实值
    val observed = testData.map(item => ((item.user, item.product), item.rating))
    val predict = predictRating.map(item => ((item.user, item.product), item.rating))
    sqrt( observed.join(predict).map{
      case ((userId, productId),(actual, pre)) =>
        val err = actual - pre
        err * err
    }.mean())
  }


}
