import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Seed for reproducibility
np.random.seed(42)

# Number of samples in the dataset
num_samples_classification = 1000
num_samples_regression = 100

# Generate synthetic data for features (soil moisture, temperature, humidity) for classification
X_classification = np.random.uniform(low=20, high=80, size=(num_samples_classification, 3))
# Generate synthetic data for target variable (crop type)
y_classification = np.random.choice(['corn', 'gram', 'cotton'], size=num_samples_classification)

# Create a DataFrame to store the classification dataset
df_classification = pd.DataFrame({
    'Soil_Moisture': X_classification[:, 0],
    'Temperature': X_classification[:, 1],
    'Humidity': X_classification[:, 2],
    'Crop_Type': y_classification
})

# Display the first few rows of the classification dataset
print(df_classification.head())

# Encode categorical target variable for classification
le_classification = LabelEncoder()
df_classification['Crop_Type'] = le_classification.fit_transform(df_classification['Crop_Type'])

# Split the classification dataset into training and testing sets
X_classification = df_classification[['Soil_Moisture', 'Temperature', 'Humidity']].values
y_classification = df_classification['Crop_Type'].values
X_train_classification, X_test_classification, y_train_classification, y_test_classification = train_test_split(
    X_classification, y_classification, test_size=0.2, random_state=42
)

# Standardize the features for classification
scaler_classification = StandardScaler()
X_train_scaled_classification = scaler_classification.fit_transform(X_train_classification)
X_test_scaled_classification = scaler_classification.transform(X_test_classification)

# Build a simple TensorFlow model for classification
model_classification = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled_classification.shape[1],)),
    tf.keras.layers.Dense(3, activation='softmax')  # Output layer for multi-class classification
])

# Compile the classification model
model_classification.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the classification model
model_classification.fit(
    X_train_scaled_classification,
    y_train_classification,
    epochs=50,
    batch_size=32,
    validation_data=(X_test_scaled_classification, y_test_classification)
)

# Ask user for input for classification and regression
temperature_input = float(input("Enter temperature: "))
soil_moisture_input = float(input("Enter soil moisture: "))
humidity_input = float(input("Enter humidity: "))
land_area_input_regression = float(input("Enter land area: "))
seed_quantity_input_regression = float(input("Enter seed quantity:"))

# Make predictions on new classification data
new_data_classification = np.array([[soil_moisture_input, temperature_input, humidity_input]])
new_data_scaled_classification = scaler_classification.transform(new_data_classification)

# Define a tf.function for prediction
@tf.function
def predict_classification(model, data):
    return model(data)

# Make predictions using the tf.function
predictions_classification = predict_classification(model_classification, new_data_scaled_classification)

predicted_class_index = np.argmax(predictions_classification, axis=1)
predicted_crop_type = le_classification.inverse_transform(predicted_class_index)[0]

# Now, use the predicted crop type to predict the yield

# Generate synthetic data for land area, seed quantity, and yield for regression
land_area_regression = np.random.uniform(low=1, high=10, size=num_samples_regression)  # in acres
seed_quantity_cotton_regression = np.random.uniform(low=1, high=2, size=num_samples_regression)  # in kg
seed_quantity_gram_regression = np.random.uniform(low=30, high=40, size=num_samples_regression)  # in kg
seed_quantity_corn_regression = np.random.uniform(low=5, high=7, size=num_samples_regression)  # in kg
yield_cotton_regression = np.random.uniform(low=200, high=800, size=num_samples_regression)  # in kg/acre
yield_gram_regression = np.random.uniform(low=1000, high=1200, size=num_samples_regression)  # in kg/acre
yield_corn_regression = np.random.uniform(low=10000, high=20000, size=num_samples_regression)  # in kg/acre

# Create separate DataFrames for seed quantity for regression
if predicted_crop_type == 'cotton':
    df_seed_regression = pd.DataFrame({
        'Land_Area': land_area_regression,
        'Seed_Quantity': seed_quantity_cotton_regression,
        'Crop_Type': 'cotton',
        'Yield_Kgs_Per_Acre': yield_cotton_regression
    })
elif predicted_crop_type == 'gram':
    df_seed_regression = pd.DataFrame({
        'Land_Area': land_area_regression,
        'Seed_Quantity': seed_quantity_gram_regression,
        'Crop_Type': 'gram',
        'Yield_Kgs_Per_Acre': yield_gram_regression
    })
else:  # 'corn'
    df_seed_regression = pd.DataFrame({
        'Land_Area': land_area_regression,
        'Seed_Quantity': seed_quantity_corn_regression,
        'Crop_Type': 'corn',
        'Yield_Kgs_Per_Acre': yield_corn_regression
    })

# Set options to display all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Display the entire DataFrame for regression
print(df_seed_regression)

# Encode categorical target variable for regression
le_regression = LabelEncoder()
df_seed_regression['Crop_Type'] = le_regression.fit_transform(df_seed_regression['Crop_Type'])

# Split the regression dataset into features and target variable
X_regression = df_seed_regression[['Land_Area', 'Seed_Quantity', 'Crop_Type']].values
y_regression = df_seed_regression['Yield_Kgs_Per_Acre'].values

# Standardize the features for regression
scaler_regression = StandardScaler()
X_scaled_regression = scaler_regression.fit_transform(X_regression)

# Build a simple TensorFlow model for regression
model_regression = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_scaled_regression.shape[1],)),
    tf.keras.layers.Dense(1)  # Output layer for regression
])

# Compile the regression model
model_regression.compile(optimizer='adam', loss='mean_squared_error')

# Train the regression model
model_regression.fit(X_scaled_regression, y_regression, epochs=50, batch_size=32)

# Make predictions on new regression data
new_data_regression = np.array([[land_area_input_regression, seed_quantity_input_regression, le_regression.transform([predicted_crop_type])[0]]])
new_data_scaled_regression = scaler_regression.transform(new_data_regression)
predicted_yield_regression = model_regression.predict(new_data_scaled_regression)[0][0] * 80
print(f'The predicted crop type is: {predicted_crop_type}')
print(f'The predicted yield is: {predicted_yield_regression} kg/acre')
