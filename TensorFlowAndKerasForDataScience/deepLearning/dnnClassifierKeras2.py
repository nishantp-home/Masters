import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras

from keras.models import Sequential
from keras.layers import Dense, InputLayer, Dropout

filePath = "E:\\Eskills-Academy-projects\\TensorFlow-and-Keras-Lecture-Data\\Data\\section9"
filePath += "\\creditcard.csv"
df = pd.read_csv(filePath)

# Check the response varaible
#df['Class'].unique()

X = df.iloc[:, :-1].values   # predictors
y = df.iloc[:, -1].values   # response

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# normalize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Define DNN Classification model

clf = Sequential([
    InputLayer(input_shape=(30,)),
    Dense(units=16, kernel_initializer='uniform', activation='relu'),
    Dense(units=18, kernel_initializer='uniform', activation='relu'),
    Dropout(0.25),
    Dense(units=20, kernel_initializer='uniform', activation='relu'),
    Dense(units=24, kernel_initializer='uniform', activation='relu'),
    Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
    ])

clf.summary()

clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
clf.fit(X_train, y_train, batch_size=15, epochs=2)

score = clf.evaluate(X_test, y_test, batch_size=128)
print("\n Score:", score[1]*100, '%')