import pandas as pd
import matplotlib.pyplot as plt

train_csv = pd.read_csv('./data/csv/train.csv')
train_image = train_csv.iloc[0:, 1:]
train_label = train_csv.iloc[0:, :1]

# train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size = 0.8, random_state = 0)

# # Display gray image (pixels value from 0 to 255)
# i = 1
# img = train_images.iloc[i].to_numpy()
# img = img.reshape((28, 28))
# plt.imshow(img, cmap = 'gray')
# plt.title(train_labels.iloc[i, 0])
# plt.hist(train_images.iloc[i])

# # Display black white image (pixels value between 0 or 1)
# i = 1
# img = train_images.iloc[i].to_numpy()
# img = img.reshape((28, 28))
# plt.imshow(img, cmap = 'binary')
# plt.title(train_labels.iloc[i, 0])
# plt.hist(train_images.iloc[i])