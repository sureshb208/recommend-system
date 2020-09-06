package com.longshine.recommender

import com.mongodb.casbah.commons.MongoDBObject
import com.mongodb.casbah.{MongoClient, MongoClientURI}
import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * 加载数据
  * 将products用^拆开：
  * 商品ID：              3982
  * 商品名称：             Fuhlen 富勒
  * 商品分类ID（不需要） ：  1057,439,736
  * 亚马逊ID（不需要） :     B009EJN4T2
  * 商品图片URL：(需要，因为需要显示）            https://images-cn-4.ssl-images-amazon.com/images/I/31QPvUDNavL._SY300_QL70_.jpg
  * 商品分类：             外设产品|鼠标|电脑/办公
  * 商品UGC标签：          富勒|鼠标|电子产品|好用|外观漂亮
  **/
// 1.分析products.csv定义商品样例类
case class Product(productId: Int, name: String, imageUrl: String, category: String, tags: String)

// 2.分析ratings.csv定义评分样例类
/**
  * 4867        :用户id
  * 457976      ：商品id
  * 5.0         ：评分
  * 1395676800  ：时间戳
  */
case class Rating(userID: Int, productId: Int, score: Double, timestamp: Long)

// 3.因为要保存到mogodb（这里不用mysql了），定义mogodb配置
case class MogoConfig(uri: String, db: String)

/**
  * 不用class 定义，是因为该任务执行一次就完事了，不用考虑代码复用
  */
object DataLoader {
  // 7.定义路径
  val PRODUCT_DATA_PATH = "/Users/lianghuikun/indigo/recommend-system/recommender/data-load/src/main/resources/products.csv"
  val RATING_DATA_PATH = "/Users/lianghuikun/indigo/recommend-system/recommender/data-load/src/main/resources/ratings.csv"
  // 12.定义mongodb中存储的表名
  val MONGODB_PRODUCT_COLLECTION = "product"
  val MONGODB_RATING_COLLECTION = "rating"


  def main(args: Array[String]): Unit = {
    // 4.定义配置
    val config = Map(
      "spark.cores" -> "local[*]",
      "mongo.uri" -> "mongodb://192.168.0.111:27017/recommender",
      "mongo.db" -> "recommender"
    )
    // 5.创建spark config
    val sparkConf = new SparkConf().setMaster(config("spark.cores")).setAppName("DataLoader")
    val spark = SparkSession.builder().config(sparkConf).getOrCreate()
    // 6.加载数据
    import spark.implicits._
    // 8.加载商品
    val productRDD = spark.sparkContext.textFile(PRODUCT_DATA_PATH)
    // 9.保存到mongo，所以转为datafame
    val productDF = productRDD.map(item => {
      val attr = item.split("\\^")
      // 转为product
      Product(attr(0).toInt, attr(1).trim, attr(4).trim, attr(5).trim, attr(6).trim)
    }).toDF()
    // 10.加载评分，并保存到mongodb
    val ratingRDD = spark.sparkContext.textFile(RATING_DATA_PATH)
    val ratingDF = ratingRDD.map(item => {
      val attr = item.split("\\,")
      Rating(attr(0).toInt, attr(1).toInt, attr(2).toDouble, attr(3).toLong)
    }).toDF()
    // 11.保存
    implicit val mongoConfig = MogoConfig(config("mongo.uri"), config("mongo.db"))
    storeDataInMongoDB(productDF, ratingDF)
    //    spark.stop()
  }

  // 13.保存到mongodb
  def storeDataInMongoDB(productDF: DataFrame, ratingDF: DataFrame)(implicit mongoConfig: MogoConfig): Unit = {
    // 13.1新建连接
    val mongoClient = MongoClient(MongoClientURI(mongoConfig.uri))
    // 13.2定义要操作的mogodb的表,可以理解为db.product
    val productColleciton = mongoClient(mongoConfig.db)(MONGODB_PRODUCT_COLLECTION)
    val ratingColleciton = mongoClient(mongoConfig.db)(MONGODB_RATING_COLLECTION)
    // 如果表已存在，则删除
    productColleciton.dropCollection()
    ratingColleciton.dropCollection()
    // 存入数据
    productDF.write
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_PRODUCT_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()
    ratingDF.write
      .option("uri", mongoConfig.uri)
      .option("collection", MONGODB_RATING_COLLECTION)
      .mode("overwrite")
      .format("com.mongodb.spark.sql")
      .save()
    // 对表创建索引
    productColleciton.createIndex(MongoDBObject("productId" -> 1))
    ratingColleciton.createIndex(MongoDBObject("userID" -> 1))
    ratingColleciton.createIndex(MongoDBObject("productId" -> 1))
  }
}
