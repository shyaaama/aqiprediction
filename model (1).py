import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import gzip

# Load the dataset'
data = pd.read_csv('output2.csv')


# Define columns to log transform and scale
columns_to_transform = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx',  'SO2', 'CO', 'Toluene']
columns_to_scale = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx',   'SO2', 'CO',  'Toluene']
categorical_features = ['City']

# Separate features and target variable
target_column = 'AQI'
X = data.drop(target_column, axis=1)
y = data[target_column]

# Log transformer function
log_transformer = FunctionTransformer(np.log1p, validate=True)

# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('log_and_scale', Pipeline(steps=[
            ('log', log_transformer),
            ('scale', StandardScaler())
        ]), columns_to_transform),
        ('scale_only', StandardScaler(), [col for col in columns_to_scale if col not in columns_to_transform]),
        ('cat', OneHotEncoder(sparse_output=False), categorical_features)
    ]
)

# Create the pipeline


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=1,
        max_features='log2',
        max_depth=20,
        bootstrap=True
    ))
])


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
print(X_train.columns)
# Fit the model
pipeline.fit(X_train, y_train)

# Predict on the test set to evaluate the model
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}, R2 Score: {r2:.2f}")
print("random forest Regressor Test Set MSE:", mse)
with open('pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

with open('pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Save the pipeline to a gzip-compressed file
with gzip.open('pipeline.pkl.gz', 'wb') as file:
    pickle.dump(pipeline, file)


print("Pipeline saved to pipeline.pkl")


