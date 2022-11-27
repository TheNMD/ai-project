import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

csv = pd.read_csv('./data/csv/train.csv')
train_image = csv.iloc[0:10000, 1:] # Form an array from row 0 to row 9999 rows, from column 1 to column 784
test_image = csv.iloc[10000:20000, 1:]
train_label = csv.iloc[0:10000, :1] # Form an array from row 0 to row 9999 rows, with column 0
test_label = csv.iloc[10000:20000, :1]

idx = 1
digit_image = train_image.iloc[idx].to_numpy() # .iloc[idx]: Pick out the idx-th row | to.numpy: transfer into an array
digit_image = digit_image.reshape((28, 28)) # Reshape into a 28x28 2d array
# Plot a digit image
plt.imshow(digit_image, cmap='gray') # Gray image
plt.savefig("./test/digit_grey.jpg")
plt.imshow(digit_image, cmap='binary') # Black white
plt.savefig("./test/digit_bw.jpg")
# Plot a digit histogram
figure(figsize=(12, 8), dpi=80)
plt.hist(train_image.iloc[idx], bins=5, edgecolor="red")
plt.title(f"Digit {train_image.iloc[idx, 0]}", fontweight="bold")
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig("./test/histogram.jpg")