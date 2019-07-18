"""
Is there a relationship between the daily minimum and maximum temperature? 
Can you predict the maximum temperature given the minimum temperature?
"""""
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("Datasets/Summary of Weather.csv")

# Step 1 -  Get the X-axis and Y-axis values or dependent and independent values
X = data["MinTemp"].values
Y = data["MaxTemp"].values

# Step 2 - Calculate the mean of the these variables
mean_X = data["MinTemp"].mean()
mean_Y = data["MaxTemp"].mean()

# Step 3 - calculate X^2, Y^2, XY

X_Sqr = [val_x * val_x for val_x in X]
Y_Sqr = [val_y * val_y for val_y in Y]
XY = [X[value] * Y[value] for value in range(len(X))]
# print("X^2 = ", X_Sqr, "\nY^2 = ", Y_Sqr, "\nXY = ", XY)

# compute the value of m and c
""" m = ((n*sum(x*y))-(sum(x)*sum(y)) / ((n*sum(x^2))-(sum(x)^2)
    c = ((sum(y)*sum(x^2))-(sum(x)*sum(x*y) / ((n*sum(x^2))-(sum(x)^2)"""

num_of_rows = data.shape[0]
sum_of_XY = sum(XY)
sum_of_X_Sqr = sum(X_Sqr)
sum_of_Y_Sqr = sum(Y_Sqr)
sum_of_X = sum(X)
sum_of_Y = sum(Y)
# print(sum_of_XY, sum_of_X_Sqr, sum_of_Y_Sqr, sum_of_X, sum_of_Y)

numerator_of_slope = ((num_of_rows * sum_of_XY) - (sum_of_X * sum_of_Y))
denominator_of_slope = ((num_of_rows * sum_of_X_Sqr) - (sum_of_X ** 2))

slope = numerator_of_slope / denominator_of_slope
# print("The value of c is : ", slope)

numerator_of_intercept = ((sum_of_Y * sum_of_X_Sqr) - (sum_of_X * sum_of_XY))
denominator_of_intercept = ((num_of_rows * sum_of_X_Sqr) - (sum_of_X * sum_of_X))

intercept = numerator_of_intercept / denominator_of_intercept
# print("The value of c is : ", intercept)

# printing 'm' and 'c'
print(f"slope= {slope} and intercept = {intercept}")
print(f"The line equation becomes y = {slope}*x + {intercept}")

# Step 4 - Plotting a regression line

# scaling the scattered data to small 2D plot
max_x, max_y = np.max(X)+100, np.max(Y)-100
# xmin, xmax, ymin, ymax = plt.axis([0, 6, 0, 6])
# getting multiple points on x(independent variable) to get the corresponding values of y(dependent variable)
x = np.linspace(max_x, max_y)

# calculating the corresponding values for y
y = slope*x + intercept
# print(y)

plt.scatter(X, Y, c='#FA8072', label='Data Given')
plt.plot(x, y, c='b', label='Regression Line')
plt.xlabel("Min Temp")
plt.ylabel("Max Temp")
plt.legend()
plt.show()

"""Using R-Squared method to check how close is our model to actual data"""

# Step 1 - Now predict some values by the now traced line y = mx+c
"""
R-squared = (sum(Y_predicted - Y_Mean))**2/ (sum(Actual Y - Y_Mean))**2
"""

rsq_numerator, rsq_denominator = 0, 0
y_pred_array = []
for i in range(num_of_rows):
    y_pred = slope*X[i] + intercept
    y_pred_array.append(y_pred)
    rsq_numerator += (y_pred-mean_Y)**2
    rsq_denominator += (Y[i]-mean_Y)**2

# print(y_pred_array, rsq_numerator, rsq_denominator)

print(f"R-Squared value of given model is {rsq_numerator/rsq_denominator}")

# calculating how much %age of predicted model matches with the actual given data
"""
% of correctness =        mean of predicted values
                    ---------------------------------------- * 100
                    mean of  dependent variable given values  
"""

percentage_of_accuracy = ((sum(y_pred_array)/len(y_pred_array)) / mean_Y) * 100

print(f"The correctness of the model is {percentage_of_accuracy}%")

warnings.simplefilter(action='ignore', category=FutureWarning)
# print(data.head())
# print(data[["MinTemp", "MaxTemp", "Date"]][data["MEA"] == 76])
# print(data.where("MinTemp" == 22.22222222))
# print(data.dtypes)

# mask = (data['birth_date'] > start_date) & (df['birth_date'] <= end_date)

if __name__ == "__main__":
    dates = data["Date"]
    print(type(dates))
    max_temp = data["MaxTemp"]
    min_temp = data["MinTemp"]
    min_temp_list, max_temp = list(min_temp), list(max_temp)
    date_list = list(dates)
    print("This model predicts the maximum temperature of the day for a given minimum temperature")
    start_date, end_date = "1942-7-1", "1945-12-31"
    input_date = input(f"Provide the date between {start_date, end_date} for which you want to check the "
                       "predicted value and actual value in format (yyyy-mm-dd)")

    if input_date in date_list:
        index_of_date = date_list.index(input_date)
        min_temperature = min_temp_list.__getitem__(index_of_date)
        print(f"The actual value of the max temp is {max_temp[index_of_date]}")
        print(f"The predicted value of the max temp is {slope*min_temperature + intercept}")
        print(f"The difference between predicted and actual value is "
              f"{(slope*min_temperature + intercept)-max_temp[index_of_date]}")
    else:
        print("You have entered an invalid or out of range date")
