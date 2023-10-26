import pickle 
import numpy as np
from util import getData
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    X, y = getData()
    Ntrain = int(len(y)/4)
    Xtrain, ytrain = X[:Ntrain], y[:Ntrain]

    model = RandomForestClassifier()
    model.fit(Xtrain, ytrain)

    Xtest, ytest = X[Ntrain:], y[Ntrain:]
    print("Test accuracy:", model.score(Xtest, ytest))

    with open('mymodel.pkl', 'wb') as f:
        pickle.dump(model, f)

