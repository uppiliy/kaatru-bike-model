# kaatru_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

def train_model():
    # Read Data
    data = pd.read_csv('Kaatru.csv') 

    drop_cols = ['instant', 'dteday', 'casual', 'registered']
    df = data.drop(columns=drop_cols)

    numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
    categorical_features = ['season', 'yr', 'mnth', 'weekday', 'weathersit']

    X = df.drop(columns=['cnt'])
    y = df['cnt']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("--- Model Evaluation ---")
    print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
    print("R²:  ", r2_score(y_test, y_pred))
    print("MAE: ", mean_absolute_error(y_test, y_pred))

    joblib.dump(pipe, 'model.pkl')
    print("Model saved to model.pkl")

if __name__ == "__main__":
    train_model()
