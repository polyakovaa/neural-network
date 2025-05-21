import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split

def plot_history(histories, model_name):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for i, history in enumerate(histories):
        plt.plot(history.history['accuracy'], label=f'Train Fold {i+1}', linestyle='--')
        plt.plot(history.history['val_accuracy'], label=f'Val Fold {i+1}')
    plt.title(f'Model Accuracy ({model_name})')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=f'Train Fold {i+1}', linestyle='--')
        plt.plot(history.history['val_loss'], label=f'Val Fold {i+1}')
    plt.title(f'Model Loss ({model_name})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

def arch_model(model_name):
  #архитектура 1: 5 слоев, dropout
  if model_name=='1':
    model = Sequential([
      Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
      Dropout(0.3),
      Dense(128, activation='relu'),
      Dropout(0.3),
      Dense(64, activation='relu'),
      Dense(32, activation='relu'),
      Dense(1, activation='sigmoid')
    ])
  elif model_name=='2':
    #архитектура 2: 7 слоев, L2 регуляризация
    model = Sequential([
      Dense(512, activation='relu', kernel_regularizer=l2(0.005), input_shape=(X_train.shape[1],)),
      Dense(256, activation='relu', kernel_regularizer=l2(0.002)),
      Dense(128, activation='relu'),
      Dense(64, activation='relu'),
      Dense(32, activation='relu'),
      Dense(16, activation='relu'),
      Dense(1, activation='sigmoid')
  ])
  elif model_name=='3':
    #архитектура 3
    model = Sequential([
      Dense(1024, activation='relu', input_shape=(X_train.shape[1],)),
      BatchNormalization(),
      Dense(512, activation='relu'),
      BatchNormalization(),
      Dense(256, activation='relu'),
      BatchNormalization(),
      Dropout(0.3),
      Dense(128, activation='relu'),
      BatchNormalization(),
      Dropout(0.2),
      Dense(64, activation='relu'),
      BatchNormalization(),
      Dropout(0.1),
      Dense(32, activation='relu'),
      BatchNormalization(),
      Dense(16, activation='relu'),
      BatchNormalization(),
      Dense(1, activation='sigmoid')
  ])
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

def fit_evaluate(train_x, val_x, train_y, val_y, EPOCHS, BATCH_SIZE, model_name, class_weights):
    model = arch_model(model_name)
    results = model.fit(train_x, train_y,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        class_weight=class_weights,
                        verbose=1,
                        validation_data=(val_x, val_y))
    return results, model

useful_cols = [2, 3, 4, 13, 18, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 36,37]
data = np.genfromtxt('/content/drive/MyDrive/nasa/nasa.csv', delimiter=',', skip_header=1, usecols=useful_cols)
y = (np.genfromtxt('/content/drive/MyDrive/nasa/nasa.csv', delimiter=',', skip_header=1, usecols=[-1], dtype=str) == 'True').astype(int)

X = np.nan_to_num(data, nan=np.nanmean(data, axis=0))
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

indx = np.arange(len(X))
np.random.shuffle(indx)
split = int(0.8 * len(X))
X_train, y_train = X[indx[:split]], y[indx[:split]]
X_test, y_test = X[indx[split:]], y[indx[split:]]

print("Общее распределение классов:")
print(f"Доля опасных: {np.mean(y):.2%}")
print("\nРаспределение в train:")
print(f"Доля опасных: {np.mean(y_train):.2%}")
print("\nРаспределение в test:")
print(f"Доля опасных: {np.mean(y_test):.2%}")

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: class_weights[0], 1: class_weights[1]}

n_folds=3  
epochs=15 
batch_size=32  


model_history = []
for i in range(n_folds):
    print("Training on Fold: ",i+1)
    t_x, val_x, t_y, val_y = train_test_split(X_train, y_train, test_size=0.1, random_state = np.random.randint(1,100, 1)[0])
    results, model = fit_evaluate(t_x, val_x, t_y, val_y, epochs, batch_size,model_name='1', class_weights)
    model_history.append(results)

model.summary()
plot_history(model_history, 'Model 1')

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Потери на тестовом наборе: {test_loss}')
print(f'Точность на тестовом наборе: {test_accuracy}')
