"""Linear Regression calculation using the least squares method

Linear regression function: y = b0 + b1x
b1 = r(sy/sx)
b0 = y_bar - b1*x_bar
"""

import numpy as np
import sys
import matplotlib.pyplot as plt

def least_squares(x, y):
	"""Finds the regression coefficients
	
	Finds the correlation coefficient r then uses the equations:
	b1 = r(sy/sx)
	b0 = y_bar - b1*x_bar
	"""
	# Ensures the values are numpy arrays
	try:
		x = np.asarray(x)
		y = np.asarray(y)
	except Exception:
		print("You must enter numpy arrays or valid lists")
	if len(x) != len(y):
		print("Length of arrays must be equal. Program Stopping.")
		sys.exit(1)

	# Calculate the regression coefficient r

	x_minus_xbar, y_minus_ybar = x-np.mean(x), y-np.mean(y)
	r = np.sum(x_minus_xbar * y_minus_ybar) / np.sqrt(np.sum(x_minus_xbar ** 2) * np.sum(y_minus_ybar ** 2))

	# Calculate the coefficients

	b1 = r * (np.std(y) / np.std(x))

	b0 = np.mean(y) - b1 * np.mean(x)

	return b0, b1

def r_squared(x, y, b):
	"""Finds the r_squared value using the calculated regression coefficients
	
	r_sq = ∑(y - y_hat)^2 / ∑(y - y_bar)^2
	"""
	# Ensures the values are numpy arrays
	try:
		x = np.asarray(x)
		y = np.asarray(y)
	except Exception:
		print("You must enter numpy arrays or valid lists")
	if len(x) != len(y):
		print("Length of arrays must be equal. Program Stopping.")
		sys.exit(1)

	# Calculate r_sq

	y_minus_ybar_sq = (y-np.mean(y)) ** 2
	y_hat_minus_y_bar_sq = ((b[0] + b[1] * x) - np.mean(y)) ** 2
	r_sq = sum(y_hat_minus_y_bar_sq) / sum(y_minus_ybar_sq)
	return r_sq

def plot_regression_line(x, y, b):
	"""Plot regression line from regression coefficients"""
	# Ensures the values are numpy arrays
	try:
		x = np.asarray(x)
		y = np.asarray(y)
	except Exception:
		print("You must enter numpy arrays or valid lists")
	if len(x) != len(y):
		print("Length of arrays must be equal. Program Stopping.")
		sys.exit(1)

	# Plot data

	plt.scatter(x, y, color = "blue")

	regression_line = b[0] + (b[1] * x)

	plt.plot(x, regression_line, color = "red")

	plt.xlabel("x")
	plt.ylabel("y")

	plt.show()

def main():
	x = np.array([17,13,12,15,16,14,16,16,18,19])
	y = np.array([94,73,59,80,93,85,66,79,77,91])

	b = least_squares(x, y)
	r_sq = r_squared(x,y,b)
	print("Regression Coefficients: b0 = {}, b1 = {}\nCoefficient of Determination = {}".format(b[0], b[1], r_sq))
	plot_regression_line(x, y, b)

if __name__ == "__main__":
	main()
