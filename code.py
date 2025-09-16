#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 20:13:56 2024

@author: mn.
"""

import pandas as pd
import numpy as np
from unidecode import unidecode
import missingno as mn
import matplotlib.pyplot as plt
import geopandas as gpd
import os
from fuzzywuzzy import process

#loading in dataset
customers = pd.read_csv('olist_customers_dataset.csv')
geolocation = pd.read_csv('olist_geolocation_dataset.csv')
items = pd.read_csv('olist_order_items_dataset.csv')
payments = pd.read_csv('olist_order_payments_dataset.csv')
orders = pd.read_csv('olist_orders_dataset.csv')
reviews = pd.read_csv('olist_order_reviews_dataset.csv')
products = pd.read_csv('olist_products_dataset.csv')
sellers = pd.read_csv('olist_sellers_dataset.csv')
categorynametrans = pd.read_csv('product_category_name_translation.csv')


###Customer Data

#checking data
customers.head()
customers.info(verbose = True)
customers.isnull().sum()
#There are no missing values in the customer dataset. None of the data is quantitative.
#There are 99441 observations in the dataset with no null values in any variable
#There are miss-spellings in city names, so we will address this in cleaning

#creating new column recording customer history refer to a customer that has ordered more than one time
#grouping column and counts
customers['customer_counts'] = customers.groupby('customer_unique_id')['customer_id'].transform('count')

#creating customer_history column
customers['customer_history'] = customers['customer_counts'].apply(lambda x: 1 if x > 1 else 0)

#drop the customer_counts column3
customers.drop(columns=['customer_counts'], inplace=True)

#check for duplicates in a column that will use for merging
print(customers['customer_id'].duplicated().sum())
#There are no duplicates in this column, making it an appropriate key to work with

#plotting the number of customers in each city
plt.bar(customers['customer_city'].value_counts().index, customers['customer_city'].value_counts().values)
plt.show()
#Too many cities so we don't use this variables in our modeling

#Plotting the number of customers in each state
plt.bar(customers['customer_state'].value_counts().index, customers['customer_state'].value_counts().values)
plt.show()


### Item Data


#checking data
items.head()
items.info(verbose = True)
items.isnull().sum()
#There are no missing values in the items dataset.
#There are 112650 elements across 7 variables
#This is split across four IDs, one date and two numeric values

#Checking summary statistics of price and freight value to gain more understanding
items.describe()
#Here we see a very low mean and high standard deviation for price when compared to the max, indicating a significant skew towards lower values. This impacts our confidence in models that assume a normal distribution
#We also see a similar sort of skew when analysing the freight value but scaled down, further diminishing our confidence in models that require a normal distribution

##creating a new column called total_items which counted the number of items of each order_id
items['total_items'] = items.groupby('order_id')['order_item_id'].transform('max')

#check for duplicates in a column that will use for merging
print(items['order_id'].duplicated().sum())
#There are 13984 duplicates, view duplicated rows 
items_duplicates  = items[items.duplicated(subset=['order_id'], keep=False)]
# These duplicate order IDs refer to orders with many products as there is one entry per product in each order

#converting shipping_limit_date to a datetime variable
items['shipping_limit_date'] = pd.to_datetime(items['shipping_limit_date'],errors='coerce')

#the data frame shows there are duplicates in order_id, product_id, seller_id but different order_item_id
#it can be assume that they are in same order but order_item_id are not grouped together
#this duplicates can be solve by group order together 
items = items.groupby('order_id').agg({
    'price': 'sum',                                     # sum prices for each order
    'freight_value': 'sum',                             # sum freight values
    'total_items': 'max',                               # max total items
    'product_id': lambda x: ', '.join(sorted(set(x))),  # unique product id
    'seller_id': lambda x: ', '.join(sorted(set(x))),   # unique seller id
    'shipping_limit_date': 'max'                        # latest shipping limit date
}).reset_index()

#check for duplicates in a column 
print(items['seller_id'].duplicated().sum())
print(items['product_id'].duplicated().sum())
#There are 94544 duplicates in seller_id
#There are 64765 duplicates in product_id
#this duplicates mean that some different order may order with same seller or same product.

#plotting a histogram of item prices
plt.hist(items['price'], bins = 500)
plt.show()
plt.hist(items['price'], bins = 100, range = [0,2000])
plt.show()
# This indicates that even at a range with fewer outliers there is a significant skew to lower prices.

#plotting a histogram of freight values
plt.hist(items['freight_value'], bins = 400)
plt.show()



### Payment Data


#checking data
payments.head()
payments.info(verbose = True)
payments.isnull().sum()
#We can see here that there are 103886 variables here across 5 variables with no null values
#3 of the variables are quantitative and 2 are qualitative
#There are quantitative data. Checking their mean, sd, max, min
payments.describe()
#Here we see thatpayment value looks to have similar distributional concerns as price before it
#we can see the same idea of low means with high standard deviations and maxes in the other two variables



#checking for duplicates in a column that will be used for merging
print(payments['order_id'].duplicated().sum())
#There are 4446 duplications of order_id that should be remove as each order_id is unique
#view which rows are duplicated
payments_duplicates  = payments[payments.duplicated(subset=['order_id'], keep=False)]

#the duplicated order_id are because a customer paid more than one payment type. 
#Therefore, for numeric column which are payments_sequential, payment_installment, and payment_value the duplication can be merged into one order_id and the data in those columns is a total sum of all payment type
#for payment_type which is non-numeric
payments = payments.groupby('order_id').agg({
    'payment_sequential': 'sum',    #sum of payment sequences
    'payment_type': ' '.join,       #concatenates the payment_type strings with a space between them
    'payment_installments': 'sum',  #sum of installments
    'payment_value': 'sum'          #sum of payment values
}).reset_index()


# Plotting a histogram of payment installments
plt.hist(payments['payment_installments'], bins = 20)
plt.show()
# We see a major cluster in single period payments, while above that we see a fairly linear reduction in payment installments except for a further cluster at the 10 month mark.

# Plotting a histogram of payment values
plt.hist(payments['payment_value'], bins = 50, range = [0,2000])
plt.show()
# Strong skew in payment value mirrors the skew in price seen in previous charts.

# Plotting a scatter plot of payment installments vs. payment values
plt.scatter(x = payments['payment_installments'], y = payments['payment_value'])
plt.show()
# Here we see that there is a huge clustering of values around single payment installments. Above this, we see no real correaltion between payment value and payment installments. This would indicate that  We also see a major cutoff around the 10 month mark, with many very low payment values with very high payment installments.



### Review Data


#checking data
reviews.head()
reviews.info(verbose = True)
reviews.isnull().sum()
#There are 89999 observations across 7 variables here, consisting of one qunatitative datatype, three qualitative datatypes and two dates
#There are 79404 missing values in review_comment_title and 52429 missing values in review_comment_message

#There are quantitative data. Checking their mean, sd, max, min
reviews.describe()
#The mean review score is exceptionally high, suggesting that there is a high prevalence of 5 and 4 star reviews that overwhelms the lower reviews
#This is supported by our visualisation highlighting the frequency of each review type (Tableau visualisation) that shows that 4 and 5 star reviews make up the bulk of the data

#creating new column for people who has historical review data
reviews['historical_review'] = reviews.apply(lambda row: 1 if pd.notna(row['review_comment_title']) 
                                   or pd.notna(row['review_comment_message']) else 0, axis=1)

#drop column review_comment_title and review_comment_message as there is not enough data to be meaningful
reviews = reviews.drop(columns=['review_comment_title', 'review_comment_message'])

#check for duplicates in a column that will use for merging
print(reviews['order_id'].duplicated().sum())
#There are 447 duplication of order_id that should be remove as each order_id is unique

#check for duplicates in review_id. Eventhough this column will not use for merging, They are unique and should not have duplication.
print(reviews['review_id'].duplicated().sum())
#There are 677 duplication of review_id that should be remove as each review_id is unique

#view which rows are duplicated
reviews_order_duplicates  = reviews[reviews.duplicated(subset=['order_id'], keep=False)]
reviews_duplicates = reviews[reviews.duplicated(subset=['review_id'], keep=False)]

#The duplicates in order_id seem to occur when someone edits their review
#keep the latest review_answer_timestamp
#convert dates to datetime
reviews['review_answer_timestamp'] = pd.to_datetime(reviews['review_answer_timestamp'], format='%d/%m/%Y %H:%M')

#sort review_answer_timestamp to keep the latest reviews answer
reviews_sorted = reviews.sort_values(by='review_answer_timestamp')

# drop duplicates, keeping only the last review for each order_id
reviews = reviews_sorted.drop_duplicates(subset='order_id', keep='last')

#The duplicates in review_id may assume that customers edited their review after using product
#keep the latest review_answer_timestamp
# drop duplicates, keeping only the last review for each review_id
reviews = reviews.drop_duplicates(subset='review_id', keep='last')


### Order Data


#checking data
orders.head()
orders.info(verbose = True)
orders.isnull().sum()
#There are 99441 values here across 8 variables, three of which are qualitative and 4 of which are dates
#There are 160 missing values in order_approved_at, 1783 in order_delivered_carrier_date, and 2965 in order_delivered_customer_date
#These columns are not use for merging. Therefore, they should be retained at this stage and not removed at this time to ensure data integrity and future usability.

#There are quantitative data. Checking their mean, sd, max, min
orders.describe()

#check for duplication in a column that will use for merging
print(orders['order_id'].duplicated().sum())
#There is no duplication in order_id 


### Product Data


#checking data
products.head()
products.info(verbose = True)
products.isnull().sum()
#There are 32951 observations here across 9 variables, two qualitative values and seven quantitative values
#There are 610 missing values in product_category_name, product_name_lenght, product_description_lenght and product_photos_qty and only two missing values in product weight, height, length and width
#There are no missing values in product_id
#product_id is a column that use for merging. The rest columns should be retained at this stage and not removed at this time to ensure data integrity and future usability.

#There are quantitative data. Checking their mean, sd, max, min
products.describe()
#Similar issues as we have seen previouslt in nearly all variables - High standard deviation, low mean and high outliers
#This seems to be less problematic for our product name lenght and dimension measurements

#check for duplicattion in a column that will use for merging
print(products['product_id'].duplicated().sum())
#There are no duplicate values in product_id

### Seller Data


#checking data
sellers.head()
sellers.info(verbose = True)
sellers.isnull().sum()
#There are 3095 observations across 4 columns
#There are no missing values in the seller dataset.

#check for duplication in a column that will use for merging
print(sellers['seller_id'].duplicated().sum())
#There are no duplicate values 

# Group by state and count the number of sellers in descending order
state_counts = sellers.groupby('seller_state').size().reset_index(name='counts')
sorted_state_counts = state_counts.sort_values(by='counts', ascending=False)
top_10_states = sorted_state_counts.head(10)
top_10_states
# Sellers are mainly concentrated in SP,PR,MG 

### Catagory Name Translation Data


#checking data
categorynametrans.info(verbose = True)
#this dataset translating product category name from Portuguese to English

#merge categorynametrans to products
#using inner join to ensures that only records with matching product_category_name in both products and catagorynametrans will be retained in the final dataFrame
products = pd.merge(products, categorynametrans, on='product_category_name', how='inner')

#product_category_name_translation
# Replace the product_category_name column with product_category_name_english
products['product_category_name'] = products['product_category_name_english']
# Drop product_category_name_english columns 
products = products.drop(columns=['product_category_name_english'])


### Joining Dataframe
#We decided to use a combination of left and inner joins in our data dependent on whether or not we considered just the review data
#Or both the data from the existing dataset and the dataset we're joining to
#The reasoning for each of these are
#Review-order: This is an inner as any review data with no order data cannot be joined to the remaining datasets, meaning there are no meaningful insights we can get from them
#review-order-item: This is a left join as the item data is not critical to determining whether or not someone will give a good review
#review-order-customer: This is an inner join as any situation where there is review data without any customer data reflects a fundamental issue with the data - how has the review been left by nobody?
#review-order-customer-product: This is an inner join as any situation where a review is left on no product is likely to be erroneous - What are they reviewing at that point?
#review-order-customer-product-payment: This is a left join as we can reasonably understand product information being omitted from certain orders, from which we can impute or flag these values
#We did not include the seller or geolocation data. geodata couldn't be included due to the lack of a strong primary key for merging


df = pd.merge(orders, reviews, on='order_id', how='inner')
df = pd.merge(df, items, on='order_id', how='left')
df = pd.merge(df, customers, on='customer_id', how='inner')
df = pd.merge(df, products, on='product_id', how='inner')
df = pd.merge(df, payments, on='order_id', how='left')
#df = pd.merge(df, sellers, on='seller_id', how='inner')
#df = pd.merge(df, geolocation, on='customer_zip_code_prefix', how='inner')

#checking dataframe after joining
df.head()
df.info(verbose = True)
df.isnull().sum()
#This generated 84219 observations across 35 variables
#There are some missing values in order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, customer_state, product_weight_g , product_length_cm, product_height_cm, and product_width_cm   

#checking the amount of missing values
missing_values = df.isnull().sum()
missing_percentage = df.isnull().sum() / len(df) * 100
print(missing_percentage)
missing_data_summary = pd.DataFrame({
    'Total Missing Values': missing_values,
    'Percentage Missing': missing_percentage})
print(missing_data_summary)
#Those missing values will be deleted as there are small amounts compare to all of data amount.
#drop NA in those specific column
#df.dropna(subset=['order_approved_at', 'order_delivered_carrier_date', 
                  #'order_delivered_customer_date', 'customer_state', 'product_weight_g' , 
                  #'product_length_cm', 'product_height_cm', 'product_width_cm'], inplace=True)

#check for duplicates in joining columns
print(df['order_id'].duplicated().sum())     #There is no duplicates
print(df['customer_id'].duplicated().sum())  #There is no duplicates
print(df['seller_id'].duplicated().sum())    #There are 79955 duplicates
print(df['product_id'].duplicated().sum())   #There are 54964 duplicates
#this duplicates mean that some different order may order with same seller or same product.
#this duplicates should not be removed




###Feature Engineering (Creating new variables after joining data)
##Timestamps

#convert to datetime format
#order_purchase_timestamp
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'],errors='coerce')

#order_delivered_customer_date,
df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'], errors='coerce')


#time_to_deliver_to_customer
df['time_to_deliver_to_customer'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.total_seconds() / (3600 * 24)




###change dummy variables to numeric using one-hot encoding

#order_status
df = pd.get_dummies(df, columns=["order_status"], prefix=["status"], dtype='int', drop_first=True)
print(df.head())

#customer_state
df = pd.get_dummies(df, columns=["customer_state"], prefix=["state"], dtype='int', drop_first=True)


###Frequency encoding variables that cannot be one hot encoded
category_freq = df['product_category_name'].value_counts(normalize=True)
df['product_category_name_freq'] = df['product_category_name'].map(category_freq)


###drop no columns that will no longer be used
df = df.drop(columns=['order_delivered_carrier_date','shipping_limit_date','order_purchase_timestamp',
                      'order_approved_at','order_delivered_carrier_date','order_delivered_customer_date',
                      'order_estimated_delivery_date','order_estimated_delivery_date','review_creation_date',
                      'review_answer_timestamp','shipping_limit_date','order_id','product_id',
                      'customer_id','customer_unique_id','seller_id','review_id','customer_city',
                      'product_category_name','customer_zip_code_prefix','payment_type','product_name_lenght', 
                      'product_description_lenght', 'product_length_cm', 
                      'product_height_cm', 'product_width_cm','payment_sequential','historical_review'])

#drop NA in those specific column
df.dropna(subset=['product_weight_g','product_photos_qty','time_to_deliver_to_customer',
                  'product_category_name_freq','payment_installments','payment_value'], inplace=True)



###Normalising Data(for numeric variables)
numeric_columns = ['product_photos_qty','product_weight_g','payment_installments',
    'price', 'freight_value', 'time_to_deliver_to_customer']

for column in numeric_columns:
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
print("Min/max normalised")
print(df)
print("\n")

###Target variable (Y) and variables (X) 

df['positive_review'] = np.where(df['review_score'] >=4, 1, 0) 
df = df.drop(columns=['review_score'])

feature_cols = df.columns
feature_cols = feature_cols.drop('positive_review')


X = df[feature_cols]
Y = df['positive_review'] # set the y
Y = np.ravel(Y) # change to an array (list)

X = df.drop('positive_review', axis=1) # drop the y from the dataframe

#Class Imbalance
proportion_of_ones = df['positive_review'].mean()
print(f"The proportion of 1s in the positive_review column is {proportion_of_ones:.2%}.")
#The proportion of 1s in the positive_review column is 79.47%, which means there are much more positive reviews than negative reviews, leading to class imbalance


#### Test-train splitting
# split data into training and test
from sklearn.model_selection  import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2024, stratify=Y)
#large data set considered using common small test size 0.2, random_state ensure consistent results across runs 4567 is random number


from sklearn.linear_model import LogisticRegression as LogR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBDT
from xgboost import XGBClassifier as XGB
from sklearn.metrics import precision_recall_fscore_support

#Logistic regression
LogR_algo = LogR()
LogR_model = LogR_algo.fit(X_train, Y_train)

#Random Forest
RF_algo = RF()
RF_model = RF_algo.fit(X_train, Y_train)

#Gradiant-Boosted Decision Trees
GBDT_algo = GBDT()
GBDT_model = GBDT_algo.fit(X_train, Y_train)

#Extreme Gradient Boosting Decision Tree
XGB_algo = XGB()
XGB_model = XGB_algo.fit(X_train, Y_train)


models = [LogR_model, RF_model, GBDT_model, XGB_model]
names = ['Logistic Regression', 'Random Forest', 'GBDT', 'XGBDT']

for i in range(4):
  print(f"Model: {names[i]}")

  # predict based on training data
  predict = models[i].predict(X_train)

  # Calculate precision, recall, and F1-score
  precision, recall, f1_score, _ = precision_recall_fscore_support(Y_train, predict, average='macro')
  print(f"Macro Precision: {precision}")
  print(f"Macro Recall: {recall}")
  print(f"Macro F1-score: {f1_score}")
  print("\n")
  
  
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# we get a load of warnings running the code so will supress them
import warnings
warnings.filterwarnings("ignore")

# create a hyperparameter search function for re-usability
def random_search(algo, hyperparameters, X_train, Y_train):
  # do the search using 3 folds/chunks as it is large dataset reducing cv to 3 will significantly speed up the process while still providing reliable results.
  # adding n_jobs=-1  to speed up RandomizedSearchCV
  clf = RandomizedSearchCV(algo, hyperparameters, cv=3, random_state=2024,
                          scoring='precision_macro', n_iter=20, n_jobs=-1, refit=True)

  # pass the data to fit/train
  clf.fit(X_train, Y_train)

  return clf.best_params_

# Logistic Regression
LogR_tuned_parameters = {
    'solver': ['liblinear'], # only this one as it does both L1 and L2
    'C': uniform(loc=0.1, scale=19.9),  # Draw from a uniform distribution between 0.1 and 20
    'penalty': ['l1', 'l2', 'elasticnet', None]
}

LogR_best_params = random_search(LogR_algo, LogR_tuned_parameters, X_train, Y_train)


# Random Forest

RF_tuned_parameters = {
    'n_estimators': randint(50, 300), # Draw from a uniform distribution between 50 and 500 fewer trees reduce computation while maintaining good performance
    'max_depth': randint(5, 15),  # Draw from a uniform distribution between 5 and 15 for large dataset expand randint to 3, 15 to prevent overfitting reduces computation
    'min_samples_split': randint(2, 10),  # Draw from a uniform distribution between 2 and 10 for large datasets, higher values can prevent overly small splits, reducing training time.
    'max_features': ['sqrt', 'log2', None]
}

RF_best_params = random_search(RF_algo, RF_tuned_parameters, X_train, Y_train)


# GBDT
GBDT_tuned_parameters = {
    'n_estimators': randint(100, 300), # Draw from a uniform distribution between 100 and 300 fewer trees reduce computation while maintaining good performance
    'learning_rate': uniform(loc=0.01, scale=0.5),  # Draw from a uniform distribution between 0.01 and 5 narrow to improve generalization and make the model more robust
    'criterion': ['friedman_mse', 'squared_error'],
    'max_depth': randint(3, 10)  # Draw from a uniform distribution between 3 and 10 maintain generalization
}

GBDT_best_params = random_search(GBDT_algo, GBDT_tuned_parameters, X_train, Y_train)


# XGBDT
XGB_tuned_parameters = {
    'n_estimators': randint(100, 500), # Draw from a uniform distribution between 50 and 500
    # eta is learning rate
    'eta': uniform(loc=0.01, scale=0.5),  # Draw from a uniform distribution between 0.01 and 5
    # objective is the same as criterion
    'objective': ['binary:logistic', 'binary:hinge'],
    'max_depth': randint(3, 10)  # Draw from a uniform distribution between 2 and 7
}
  
XGB_best_params = random_search(XGB_algo, XGB_tuned_parameters, X_train, Y_train)



# Train the models
LogR_algo = LogR(**LogR_best_params)
LogR_model = LogR_algo.fit(X_train, Y_train)

RF_algo = RF(**RF_best_params)
RF_model = RF_algo.fit(X_train, Y_train)

GBDT_algo = GBDT(**GBDT_best_params)
GBDT_model = GBDT_algo.fit(X_train, Y_train)

XGB_algo = XGB(**XGB_best_params)
XGB_model = XGB_algo.fit(X_train, Y_train)


# score the models
models = [LogR_model, RF_model, GBDT_model, XGB_model] # redo this or it uses the old models

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

for i in range(4):
  print(f"Model: {names[i]}")

  # predict based on training data
  predict = models[i].predict(X_train)

  accuracy = accuracy_score(Y_train, predict)
  
  # Calculate precision, recall, and F1-score
  precision, recall, f1_score, _ = precision_recall_fscore_support(Y_train, predict, average='macro')
  print(f"Accuracy: {accuracy}")
  print(f"Macro Precision: {precision}")
  print(f"Macro Recall: {recall}")
  print(f"Macro F1-score: {f1_score}")
  print("\n")

for i in range(4):
  print(f"Model: {names[i]}")

  # predict based on TEST data
  predict = models[i].predict(X_test)

  accuracy = accuracy_score(Y_test, predict)
  
  # Calculate precision, recall, and F1-score
  precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, predict, average='macro')
  print(f"Accuracy: {accuracy}")
  print(f"Macro Precision: {precision}")
  print(f"Macro Recall: {recall}")
  print(f"Macro F1-score: {f1_score}")
  print("\n")

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay as CM

plt.show()

# Logistic regression
print("LogR Confusion Matrix")
predict = LogR_model.predict(X_test)
CM.from_predictions(Y_test, predict)

# Random Forest
print("Random Forest Confusion Matrix")
predict = RF_model.predict(X_test)
CM.from_predictions(Y_test, predict)

# GBDT
print("GBDT Confusion Matrix")
predict = GBDT_model.predict(X_test)
print(CM.from_predictions(Y_test, predict))

# XGB
print("XGB Confusion Matrix")
predict = XGB_model.predict(X_test)
print(CM.from_predictions(Y_test, predict))


















