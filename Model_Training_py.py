import pandas as pd
import numpy as np

df=pd.read_csv('Fish.csv')

df

# Independent and Dependent features
X=df.iloc[:,:-1]
y=df.iloc[:,-1]

X.head()

y.head(20)

# Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=7)

# Initialize the instance of Gradient Boost Classifier
from sklearn.ensemble import GradientBoostingClassifier

gb_clf = GradientBoostingClassifier()
# Use all the defaults of the model as set in sci-kit learn
# GradientBoostingClassifier(*, loss='log_loss', learning_rate=0.1, 
#                            n_estimators=100, subsample=1.0, 
#                            criterion='friedman_mse', min_samples_split=2, 
#                            min_samples_leaf=1, min_weight_fraction_leaf=0.0, 
#                            max_depth=3, min_impurity_decrease=0.0, init=None, 
#                            random_state=None, max_features=None, verbose=0, 
#                            max_leaf_nodes=None, warm_start=False, 
#                            validation_fraction=0.1, n_iter_no_change=None, 
#                            tol=0.0001, ccp_alpha=0.0)
# Fit the training data to the gradient boost model
gb_clf.fit(X_train, y_train)

# Prediction
y_pred=gb_clf.predict(X_test)

# Check Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)

score

# Create a Pickle file  
import pickle
pickle_out = open("gb_clf.pkl","wb")
pickle.dump(gb_clf, pickle_out)
pickle_out.close()

# gb_clf.predict([[110,19.1,20.8,23.1,6.1677,3.3957]])

pickle_in = open("gb_clf.pkl","rb")
classifier=pickle.load(pickle_in)
print(classifier.predict([[110,19.1,20.8,23.1,6.1677,3.3957]]))

