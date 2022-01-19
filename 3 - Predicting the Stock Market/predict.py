import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Read the CSV into stocks
stocks = pd.read_csv('sphist.csv')
stocks['Date'] = pd.to_datetime(stocks['Date'])
stocks.sort_values(by=['Date'], ascending=False, inplace=True)

# Calculating the 3 indicators to add to stocks (shifting over by 1)
stocks['past5']  = stocks['Close'].rolling(5).mean().shift()
stocks['past30'] = stocks['Close'].rolling(30).mean().shift()
stocks['past365'] = stocks['Close'].rolling(365).mean().shift()

# Drop those rows without historical data
stocks = stocks[stocks['Date'] > datetime(year=1951, month=1, day=2)]

# Drop any NaN rows
stocks.dropna(inplace=True)

# Split them into train/test based off date
train = stocks[stocks['Date'] < datetime(year=2013, month=1, day=1)]
test = stocks[stocks['Date'] >= datetime(year=2013, month=1, day=1)]

# Train a linear regression model
lr = LinearRegression()
lr.fit(train[['past5','past30','past365']], train['Close'])

# Predict with model and check error
predictions = lr.predict(test[['past5','past30','past365']])
error = mean_absolute_error(test['Close'], predictions)

print('Mean absolute error: ' + str(error))