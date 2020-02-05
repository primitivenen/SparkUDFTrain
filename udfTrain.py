# Databricks notebook source
dbutils.fs.mkdirs("dbfs:/FileStore/temporary")

# COMMAND ----------

import joblib # To Pickel Trained model file 
import numpy as np # To create random data
import pandas as pd # To operate on data in Python Process
from sklearn.linear_model import LinearRegression # To train Linear Regression models

from pyspark.sql.functions import pandas_udf, PandasUDFType # Pandas UDF functions to call Python processes from spark
from pyspark.sql.types import DoubleType, StringType, ArrayType # Data types to capture reurn at Spark End

# COMMAND ----------

df1 = pd.DataFrame({'x': np.random.normal(size=100)})
df1['y'] = df1['x']*2.5 + np.random.normal(scale=0.5, size=100) # DF1 is dummy Linear data Y = 2.5*x + random noise of 100 datapoints
df1['name'] = 'df1'
df1.to_csv('df1.csv')

# COMMAND ----------

# MAGIC %sh
# MAGIC gzip -f "df1.csv"

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/df1.csv.gz", "dbfs:/FileStore/temporary/df1.csv.gz")

# COMMAND ----------

df2 = pd.DataFrame({'x': np.random.normal(size=100)})
df2['y'] = df2['x']*3.0 + np.random.normal(scale=0.3, size=100) # DF2 is dummy Linear data Y = 3.0*x + random noise of 100 datapoints
df2['name'] = 'df2'
df2.to_csv('df2.csv')

# COMMAND ----------

# MAGIC %sh
# MAGIC gzip -f "df2.csv"

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/df2.csv.gz", "dbfs:/FileStore/temporary/df2.csv.gz")

# COMMAND ----------

# MAGIC %fs ls dbfs:/FileStore/temporary

# COMMAND ----------

sparkDF = (spark.read
            .option("header", "true")
            .option("delimiter", ",")
            .option("inferSchema", "true") 
            .csv('dbfs:/FileStore/temporary/df*.csv.gz'))

sparkDF.rdd.getNumPartitions()

# COMMAND ----------

@pandas_udf(returnType=DoubleType())
def train_lm_pandas_udf(*cols):
    df = pd.concat(cols, axis=1) # Create pandas dataframe using input Spark DataFrame columns
    df.columns = ['x', 'y', 'name']
    modelUDF = LinearRegression() # Scikit-Learn Linear Regression 
    modelUDF.fit(pd.DataFrame(df['x']),df['y']) # Fit Scikit-Learn Linear Regression Model
    sig = df.loc[0,'name'] # Unique Identiter for model files, obtained from one of the columns in dataset
    joblib.dump(modelUDF, 'modelUDF{signature}.joblib'.format(signature=sig)) # Pickel Thetrained model file
    return pd.Series(modelUDF.predict(pd.DataFrame(df['x']))) # Returns Predicted values on training data

# COMMAND ----------

sparkDF.printSchema()

# COMMAND ----------

column_names = ['x', 'y', 'name']
sparkDF2 = sparkDF.select(train_lm_pandas_udf(*column_names).alias("TrainPrediction"))

# COMMAND ----------

sparkDF2.collect()

# COMMAND ----------

# MAGIC %fs ls file:/databricks/driver

# COMMAND ----------

modeldf1 = joblib.load('modelUDFdf1.joblib')
modeldf2 = joblib.load('modelUDFdf2.joblib')

# COMMAND ----------

modeldf1.coef_

# COMMAND ----------

modeldf2.coef_

# COMMAND ----------


