# Ex.No: 6               HOLT WINTERS METHOD
### Date: 

### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
Importing necessary modules
```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
```
Load the dataset,perform data exploration
```
data = pd.read_csv('AirPassengers.csv', parse_dates=['Month'],index_col='Month')
data.head()
```
Resample and plot data
```
data_monthly = data.resample('MS').sum()   #Month start
data_monthly.head()
data_monthly.plot()
```
Scale the data and check for seasonality
```
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),index=data_monthly.index)

scaled_data.plot() # The data seems to have additive trend and multiplicative seasonality

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data_monthly, model="additive")
decomposition.plot()
plt.show()
```
Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate the model predictions against test data
```
scaled_data=scaled_data+1 # multiplicative seasonality cant handle non postive values, yes even zeros
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')

np.sqrt(mean_squared_error(test_data, test_predictions_add))

np.sqrt(scaled_data.var()),scaled_data.mean()
```
Create the final model and predict future data and plot it
```
final_model = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul', seasonal_periods=12).fit()

final_predictions = final_model.forecast(steps=int(len(data_monthly)/4)) #for next year

ax=data_monthly.plot()
final_predictions.plot(ax=ax)
ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel('Number of monthly passengers')
ax.set_ylabel('Months')
ax.set_title('Prediction')
```

### OUTPUT:
Scaled_data plot:

<img width="593" height="462" alt="image" src="https://github.com/user-attachments/assets/6d82ee6e-593f-4461-81cd-06ada3730281" />

Decomposed plot:

<img width="678" height="507" alt="image" src="https://github.com/user-attachments/assets/ec7b8c11-590e-4f99-ba1c-daa9e7cb77c2" />

Test prediction:

<img width="590" height="479" alt="image" src="https://github.com/user-attachments/assets/0b76fabd-b682-4df2-9210-1dcd1ef527ce" />

Final prediction:

<img width="621" height="523" alt="image" src="https://github.com/user-attachments/assets/e2edc728-d7f5-4e02-8437-5d6060ed2640" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
