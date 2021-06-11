from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

IMAGE_WIDTH = 40
IMAGE_HEIGHT = 40
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
differ_colors = True

df = pd.DataFrame(columns=['img', 'label'])

FEN_dict = {'black_bishop': 'b', 'black_king': 'k', 'black_knight': 'n', 'black_pawn': 'p',
            'black_queen': 'q', 'black_rook': 'r', 'white_bishop': 'B', 'white_king': 'K',
            'white_knight': 'N', 'white_pawn': 'P', 'white_queen': 'Q', 'white_rook': 'R'}

# Load data into dataframe

categories = os.listdir("../../data")

for cat in categories:
    # if 'black_pawn' in cat:
    #     continue
    for file in os.listdir('../../data/' + cat):
        image = cv2.imread(os.path.join('../../data', cat, file), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, IMAGE_SIZE)
        if differ_colors:
            label = cat
        else:
            label = cat.split('_')[1]
        df = df.append({'img': image, 'label': label}, ignore_index=True)

df = df.sample(frac=1).reset_index(drop=True)

# Process loaded data
df['img'] = [i / 255 for i in df['img']]
X = df['img'].to_numpy()
X = np.stack(X)
X = X.reshape(X.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, 1)
y = pd.get_dummies(df['label'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# Build model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
if differ_colors:
    n_last_neurons = 12
else:
    n_last_neurons = 6
model.add(Dense(n_last_neurons, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

early_stop = EarlyStopping(patience=5, monitor='val_acc', restore_best_weights=True)
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
callbacks = [early_stop]

model.fit(X_train, y_train, callbacks=callbacks, epochs=100, validation_split=.2)

# print(model.evaluate(X_test, y_test))

plt.figure(figsize=(15, 15))
print(model.evaluate(X_test, y_test)[1])
l = [i.split('_')[1] for i in categories]

for i in range(len(X_test)):
    plt.subplot(4, 19, i + 1)
    plt.imshow(X_test[i], cmap='gray')
    plt.title(FEN_dict[categories[np.argmax(model.predict(X_test[i].reshape(1, 40, 40, 1)))]])
plt.show()
