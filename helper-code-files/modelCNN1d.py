#import classes and functions
import pandas as pd
import numpy as np
np.random.seed(2500)
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, MaxPooling1D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sn

X = [] # list for payload bytes  [independent variable(s)]
Y = [] # list for application protocol [dependent varibale(s)]

# Load the dataset
df = pd.read_csv('dataset.csv', header=None)
dataArray = df.values


for data in dataArray:	
    X.append(data[1:])
    Y.append(data[0])

X = np.array(X)
Y = np.array(Y)

# Create training and testing vars
X_train, x_test, Y_train, y_test = train_test_split(
			X, Y, test_size=0.30, random_state=42)

# reshape to spatial dimensions
X_train = np.expand_dims(X_train, axis=2) 
x_test = np.expand_dims(x_test, axis=2)


# Encode the Output Variable (Y = Protocol)
encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y_train = encoder.transform(Y_train)
y_train = np_utils.to_categorical(encoded_Y_train)

encoder.fit(y_test)
encoded_y_test = encoder.transform(y_test)
class_labels = encoder.classes_	# save the names of the classes after encoding
y_test = np_utils.to_categorical(encoded_y_test)

nb_classes = len(y_train[0]) # number of output variables

# 1d conv
model = Sequential()
model.add(Convolution1D(filters=2, kernel_size=5, activation='relu', input_shape=(1024,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))
model.load_weights('model_weight.hdf5')    

sgd = SGD(lr=0.001)
model.compile(optimizer='sgd', loss='categorical_crossentropy',
				metrics=['accuracy'])
#model.fit(X_train, y_train, epochs=50, batch_size=32)   # NOT batch_size=16
model.save_weights('model_weight.hdf5')

score = model.evaluate(x_test, y_test, verbose=0)
print(score)
y_preds = model.predict(x_test, batch_size=32, verbose=0)


# prepare data for confusion matrix
y_test_non_category = [ np.argmax(t) for t in y_test ]
y_predict_non_category = [ np.argmax(t) for t in y_preds ]
cm = confusion_matrix(y_test_non_category, y_predict_non_category)


# plot confusion matrix
df_cm = pd.DataFrame(cm) 
plt.figure(figsize = (20,15))
plt.xlabel('Predicted')
plt.ylabel('True')
fig = sn.heatmap(df_cm, cmap='coolwarm', xticklabels=class_labels, 
				yticklabels=class_labels, linewidths=.5, annot=True, fmt="d")
fig.get_figure().savefig('confusion_matrix.pdf', dpi=400)
