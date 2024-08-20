import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
#cross validation
from sklearn.model_selection import cross_val_score
# Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

#select the columns for the model
X=data[['Age','Sex','Pclass','SibSp','Parch','Fare']]
y=data['Survived']

# Custom Age Imputer
class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X['Age'] = X['Age'].fillna(X['Age'].mean())
        return X
    
# Add IsChild feature
class ChildAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X['IsChild'] = (X['Age'] < 18).astype(int)
        return X
    
# Apply MinMax scaler for Age and Fare
class Scaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        X['Age'] = (X['Age'] - X['Age'].min()) / (X['Age'].max() - X['Age'].min())
        X['Fare'] = (X['Fare'] - X['Fare'].min()) / (X['Fare'].max() - X['Fare'].min())
        return X
# Custom Encoder
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X = X.copy()
        X['Sex'] = X['Sex'].map({'male': 1, 'female': 0})
        return X

# Split data into train and test  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
def fit_and_evaluate_pipeline(pipeline, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):

    pipeline.fit(X_train, y_train)
    # Evaluate the model
    y_pred = pipeline.predict(X_test)
    Y_train_pred=pipeline.predict(X_train)
    print("Train_Accuracy:", accuracy_score(y_train, Y_train_pred))
    print("Test_Accuracy:", accuracy_score(y_test, y_pred))
    print("Train_Precision:", precision_score(y_train, Y_train_pred))
    print("Test_Precision:", precision_score(y_test, y_pred))
    print("Train_Recall:", recall_score(y_train, Y_train_pred))
    print("Test_Recall:", recall_score(y_test, y_pred))
    scores = cross_val_score(pipeline, X, y, cv=5)
    print("Cross-validation scores:", scores)
    print("Mean cross-validation score:", scores.mean())

# use for random forest
RFC_Pipeline=Pipeline([
    ('age_imputer',AgeImputer()),
    ('child_adder',ChildAdder()),
    ('categorical_encoder',CategoricalEncoder()),
    ('scaler',Scaler()),
    ('random_forest',RandomForestClassifier( n_estimators=100, random_state=42, max_depth=5))

])

fit_and_evaluate_pipeline(RFC_Pipeline)
