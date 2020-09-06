package com.longshine

import org.apache.spark.SparkConf
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.sql.SparkSession
import org.jblas.DoubleMatrix

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

object OfflineRecommender {

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
        rating => (rating.userID, rating.productId, rating.score)
      ).cache()
    // 提取出所有的用户和商品的数据集
    val userRDD = ratingRDD.map(_._1).distinct()
    val productRDD = ratingRDD.map(_._2).distinct()
    // TODO: 6.核心计算过程
    // 6.1训练隐语义模型
    val trainData = ratingRDD.map(x => Rating(x._1, x._2, x._3))
    // 定义模型训练的参数，rank隐特征个数，iterations迭代次数，lambda正则化系数
    val (rank, iterations, lambda) = (5, 10, 0.01);
    val model = ALS.train(trainData, rank, iterations, lambda)
    // 6.2获得预测评分矩阵，得到用户的推荐列表
    // 为所有用户推荐商品，用userRDD和productRDD做笛卡尔乘积
    val userProducts = userRDD.cartesian(productRDD)
    val preRating = model.predict(userProducts)
    // 从预测评分矩阵中提取得到用户推荐列表
    val userRecs = preRating
      .filter(_.rating > 0) // 评分大于0
      .map(
      rating => (rating.user, (rating.product, rating.rating))
    ).groupByKey()
      .map {
        case (userId, recs) => {
          val recommendations = recs.toList
            .sortWith(_._2 > _._2) // _._2表示按(rating.product, rating.rating)中的rating评分，>_._2表示降序
            .take(USER_MAX_RECOMMENDATION) // 取前20个
            .map(x => Recommendation(x._1, x._2)) // 转为自定义的Recommendation对象
          UserRecs(userId, recommendations)
        }
      }
      .toDF() // 要保存到mongodb，因此转为dataframe
    userRecs.write
      .option("uri", mongoConfig.uri)
      .option("collection", UserRecs_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()
    // 6.3利用商品的特征向量，计算商品的相似度列表
    // 从训练好的模型中获取商品特征矩阵
    val productFeatures = model.productFeatures
      .map {
        // 转为矩阵
        case (productId, features) => (productId, new DoubleMatrix(features))
      }
    // 6.4两两配对商品，计算余弦相似度
    // 自己跟自己做笛卡尔
    val productRecs = productFeatures.cartesian(productFeatures)
      .filter {
        // 计算相似度，自己不能和自己计算相似度
        case (a, b) => a._1 != b._1
      }
      .map {
        // 计算余弦相似度
        case (a, b) => {
          val simScore = consinSim(a._2, b._2)
          // 相似度列表格式： (商品id，（商品id，评分））
          (a._1, (b._1, simScore))
        }
      }
      .filter(_._2._2 > 0.4) // 相似度大于0.4
      .groupByKey()
      .map {
        case (productId, recs) => {
          val recommendations = recs.toList
            .sortWith(_._2 > _._2) // _._2表示按(rating.product, rating.rating)中的rating评分，>_._2表示降序
            .take(USER_MAX_RECOMMENDATION) // 取前20个
            .map(x => Recommendation(x._1, x._2)) // 转为自定义的Recommendation对象
          ProductRecs(productId, recommendations)
        }
      }
      .toDF() // 要保存到mongodb，因此转为dataframe

    productRecs.write
      .option("uri", mongoConfig.uri)
      .option("collection", ProductRecs_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()

    spark.stop()
  }

  // 计算相似度
  def consinSim(product1: DoubleMatrix, product2: DoubleMatrix): Double = {
    product1.dot(product2) / (product1.norm2() * product2.norm2())
  }

}
