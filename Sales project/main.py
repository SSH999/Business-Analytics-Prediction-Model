# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the data into a pandas dataframe
df = pd.read_csv('sales_data.csv')


print("Dimensions of the data:")
print(df.shape)


print("\nMissing data:")
print(df.isnull().sum())

# I am checking for  duplicates
print("\nDuplicates:")
print(df.duplicated().sum())

#  outliers
print("\nOutliers:")
print(df.describe())

# Visualize the data using histograms, scatter plots, and line plots
plt.hist(df['sales'], bins=10)
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

plt.scatter(df['date'], df['sales'])
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

plt.plot(df['date'], df['sales'])
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()
# Remove duplicates
df.drop_duplicates(inplace=True)

# This is a precaution im using to handle missing data
df.dropna(inplace=True)

# This ensures that i have removed outliers
Q1 = df['sales'].quantile(0.25)
Q3 = df['sales'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['sales'] >= Q1 - 1.5*IQR) & (df['sales'] <= Q3 + 1.5*IQR)]
#I have  extracted month and year from date
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# now I have aggregated sales by month
df_monthly = df.groupby(['year', 'month'], as_index=False)['sales'].mean()
df_monthly['date'] = pd.to_datetime(df_monthly[['year', 'month']].assign(day=1))
# Split the data into training and testing sets
train = df_monthly[df_monthly['date'] < '2017-01-01']
test = df_monthly[df_monthly['date'] >= '2017-01-01']

# Define the independent and dependent variables
X_train = train[['month', 'year']]
y_train = train['sales']
X_test = test[['month', 'year']]
y_test = test['sales']

# Im using a linear regression model not logistic regresiion
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test set
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean squared error:", mse)
print("R-squared:", r2)
import matplotlib.pyplot as plt

# Plot the actual sales data
plt.plot(test['date'], y_test, label='Actual')

# Plot the predicted sales data
plt.plot(test['date'], y_pred, label='Predicted')

# Add labels and legend
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Actual vs. Predicted Sales')
plt.legend()

# Visulaize plot with matplotlib
plt.show()
