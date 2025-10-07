import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,root_mean_squared_error

try:
    # Load dataset
    filepath = 'D:/Content Monetization Modeler/youtube_ad_revenue_dataset.csv'
    data = pd.read_csv(filepath)
    # print("Dataset loaded successfully.")
    columns_list = list(data.columns)
 
    #Analysis of Columns
    # Treating of Missing values
    missing_df = data.isnull().sum()
    for i in columns_list:
      missing_value_column = missing_df[i].item()
      number_of_rows = data.shape[0]
      percentage_of_missing_values = (missing_value_column / number_of_rows) * 100
    #   print(i, 'Column', 'percentage_of_missing_values = ', percentage_of_missing_values)
    data['likes'] = data['likes'].fillna(data['likes'].mean())
    data['comments'] = data['comments'].fillna(data['comments'].mean())
    data = data.dropna(axis = 0,subset=['watch_time_minutes'])
    # data.info()
    # print(data.isnull().sum())
    #column wise analysis
    data['date'] = pd.to_datetime(data['date'], format='mixed')
    data['year'] = pd.to_datetime(data['date']).dt.year
    data['month'] = pd.to_datetime(data['date']).dt.month.astype('int')
    data['day'] = pd.to_datetime(data['date']).dt.day.astype('int')
    
    data['comments'] = data['comments'].astype(int)
    data['likes'] = data['likes'].astype(int)
    #FEATURE ENGINEERING
    # Calculate total engagements
    data['engagements'] = data['likes'] + data['comments']
    # Calculate Engagement Rate
    data['engagement_rate (%)'] = round((data['engagements'] / data['views']) * 100,2)

    #encoding of the categorical column
    le = LabelEncoder()
    data['device'] = le.fit_transform(data['device'])
    #mapping the nominal column
    data['country'] = data['country'].map({'US':0, 'UK':1, 'CA':2, 'AU':3, 'DE':4,'IN':5,'OTHERS':6})
    data['category'] = data['category'].map({'Entertainment':0, 'Gaming':1, 'Education':2, 'Music':3, 'Lifestyle':4, 'Tech':5,'Others':6})
   

    # dropping the columns
    # Engagement_rate is derived from engagements ,views and likes we can drop engagements and likes column since they are highly correlated.
    model_data = data.copy()
    model_data = model_data.drop(columns=['video_id','year','month','day','engagements','engagement_rate (%)','date'], axis=1)
    model_data = model_data.dropna(axis = 0)

    # train test split
    X = model_data.drop('ad_revenue_usd', axis=1)
    y = model_data['ad_revenue_usd']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # # # #scaling of the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
   
   

    # model training 1
    model1 = Ridge(alpha=10.0)
    model1.fit(X_train, y_train)
    y_train_pred1 = model1.predict(X_train)
    y_pred1 = model1.predict(X_test)


    # #model training 2
    model2 = LinearRegression()
    model2.fit(X_train,y_train)
    y_train_pred2 = model2.predict(X_train)
    y_pred2 = model2.predict(X_test)
    # MODEL TRAINING 3
    model3 = ElasticNet(alpha=0.01, l1_ratio=0.5)
    model3.fit(X_train, y_train)
    y_train_pred3 = model3.predict(X_train)
    y_pred3 = model3.predict(X_test)
    #model training 4
    model4 = Lasso(alpha=0.1)
    model4.fit(X_train, y_train)
    y_train_pred4 = model4.predict(X_train)
    y_pred4 = model4.predict(X_test)

    # model training 5
    model5 = SVR(kernel='linear', C=1.0, epsilon=0.1)
    model5.fit(X_train, y_train)
    y_train_pred5 = model5.predict(X_train)
    y_pred5 = model5.predict(X_test)


    # # # # #model evaluation
    print("Model Evaluation for Ridge:")
    print("RMSE Score on training set:", root_mean_squared_error(y_train, y_train_pred1))
    print("MSE Score on training set:", mean_squared_error(y_train, y_train_pred1))
    print("MAE Score on training set:", mean_absolute_error(y_train, y_train_pred1))
    print("r2_score on training set:", r2_score(y_train, y_train_pred1))
    print("RMSE Score:", root_mean_squared_error(y_test, y_pred1))
    print("MAE Score:", mean_absolute_error(y_test, y_pred1))
    print("MSE Score:", mean_squared_error(y_test, y_pred1))
    print("r2_score:", r2_score(y_test, y_pred1))
    print("Coefficients:", model1.coef_)
    print("Intercept:", model1.intercept_)

    # # # #model evaluation
    print("Model Evaluation for Linear Regression:")
    print("RMSE Score on training set:", root_mean_squared_error(y_train, y_train_pred2))
    print("MSE Score on training set:", mean_squared_error(y_train, y_train_pred2))
    print("MAE Score on training set:", mean_absolute_error(y_train, y_train_pred2))
    print("r2_score on training set:", r2_score(y_train, y_train_pred2))
    print("RMSE Score:", root_mean_squared_error(y_test, y_pred2))
    print("MAE Score:", mean_absolute_error(y_test, y_pred2))
    print("MSE Score:", mean_squared_error(y_test, y_pred2))
    print("r2_score:", r2_score(y_test, y_pred2))
    print("Coefficients:", model2.coef_)
    print("Intercept:", model2.intercept_)

    print("Model Evaluation for ElasticNet:")
    print("RMSE Score on training set:", root_mean_squared_error(y_train, y_train_pred3))
    print("MSE Score on training set:", mean_squared_error(y_train, y_train_pred3))
    print("MAE Score on training set:", mean_absolute_error(y_train, y_train_pred3))
    print("r2_score on training set:", r2_score(y_train, y_train_pred3))
    print("RMSE Score:", root_mean_squared_error(y_test, y_pred3))
    print("MAE Score:", mean_absolute_error(y_test, y_pred3))
    print("MSE Score:", mean_squared_error(y_test, y_pred3))
    print("r2_score:", r2_score(y_test, y_pred3))
    print("Coefficients:", model3.coef_)
    print("Intercept:", model3.intercept_)


    print("Model Evaluation for SVR:")
    print("RMSE Score on training set:", root_mean_squared_error(y_train, y_train_pred4))
    print("MSE Score on training set:", mean_squared_error(y_train, y_train_pred4))
    print("MAE Score on training set:", mean_absolute_error(y_train, y_train_pred4))
    print("r2_score on training set:", r2_score(y_train, y_train_pred4))
    print("RMSE Score:", root_mean_squared_error(y_test, y_pred4))
    print("MAE Score:", mean_absolute_error(y_test, y_pred4))
    print("MSE Score:", mean_squared_error(y_test, y_pred4))
    print("r2_score:", r2_score(y_test, y_pred4))
    print("Coefficients:", model4.coef_)
    print("Intercept:", model4.intercept_)

    print("Model Evaluation for Lasso:")
    print("RMSE Score on training set:", root_mean_squared_error(y_train, y_train_pred5))
    print("MSE Score on training set:", mean_squared_error(y_train, y_train_pred5))
    print("MAE Score on training set:", mean_absolute_error(y_train, y_train_pred5))
    print("r2_score on training set:", r2_score(y_train, y_train_pred5))
    print("RMSE Score:", root_mean_squared_error(y_test, y_pred5))
    print("MAE Score:", mean_absolute_error(y_test, y_pred5))
    print("MSE Score:", mean_squared_error(y_test, y_pred5))
    print("r2_score:", r2_score(y_test, y_pred5))
    print("Coefficients:", model5.coef_)
    print("Intercept:", model5.intercept_)
    
    #Freezing ridge model for the application so pickling the file
    ridge_model = model1
    with open('ridge_model.pkl','wb') as file:
      pickle.dump(ridge_model,file)
    feature_names = X.columns.to_list()
    with open('feature_names.pkl','wb') as file1:
      pickle.dump(feature_names,file1)
    with open("scaler.pkl", "wb") as f:
      pickle.dump(scaler, f)
except Exception as e:
   print("Error",e)








