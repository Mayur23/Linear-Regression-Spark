import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// To see less warninngs and output
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Clean-USA-Housing.csv")

data.printSchema()

// See an example of what the data looks like
// by printing out a Row
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example Data Row")

//To show a better representation of the dataset than what "head" command shows,  grab the first row in the column
//and print out the column followed by the value
for(ind <- Range(1,colnames.length)){
  println(colnames(ind))
  println(firstrow(ind))
  println("\n")
}


// Rename the Price column as Label since we want to predict the price column. Grab all the numerical columns too  
val df = data.select(data("Price").as("label"),$"Avg Area Income",$"Avg Area House Age",$"Avg Area Number of Rooms",$"Area Population")

//Use vector assembler to create the Data Frame. Pass an array of i/p columns you expect to receive 
val assembler = new VectorAssembler().setInputCols(Array("Avg Area Income","Avg Area House Age","Avg Area Number of Rooms","Area Population")).setOutputCol("features")

//Take all the columns in the data frame and put them into an array an of single column called features
val output = assembler.transform(df).select($"label",$"features")

//Create a Linear Regression model 
val lr = new LinearRegression()

//Fit the model to the data 
val lrModel = lr.fit(output)

//Print the coefficients and intercepts for Linear Regression 
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

//Summarize the model over training sets and print metrics
val trainingSummary = lrModel.summary

println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")


trainingSummary.residuals.show() //Predictions,r2, and more

println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"r2: ${trainingSummary.r2}")
