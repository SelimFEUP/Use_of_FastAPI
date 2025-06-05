import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout    
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    return data.values

path = 'transformed_data.csv'
data = load_data(path)
print(data.shape)

def create_seq(data, input_steps, output_steps):
    x, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        x.append(data[i:(i+input_steps), :])
        y.append(data[(i+input_steps):(i+input_steps+output_steps), :])
    return np.array(x), np.array(y)
x, y = create_seq(data, 24, 24)
print(x.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape)


scaler = MinMaxScaler()
num_feature = x_train.shape[2]
x_train_reshaped = x_train.reshape(-1, num_feature)
x_test_reshaped = x_test.reshape(-1, num_feature)
x_train_scaled = scaler.fit_transform(x_train_reshaped).reshape(x_train.shape)
x_test_scaled = scaler.transform(x_test_reshaped).reshape(x_test.shape)
y_train_reshaped = y_train.reshape(-1, num_feature)
y_test_reshaped = y_test.reshape(-1, num_feature)
y_train_scaled = scaler.fit_transform(y_train_reshaped).reshape(y_train.shape)
y_test_scaled = scaler.transform(y_test_reshaped).reshape(y_test.shape) 

print(x_train_scaled.shape, y_train_scaled.shape)
print(x_test_scaled.shape, y_test_scaled.shape)


def create_model(input_shape):
    input_layer = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.DepthwiseConv1D(kernel_size=3, padding='same')(input_layer)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(64, return_sequences=True)(x)  # <-- keep return_sequences=True
    x = Dropout(0.2)(x)
    output_layer = Dense(input_shape[1])(x)  # output shape: (batch, 24, num_feature)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model  
    

input_shape = (x_train_scaled.shape[1], x_train_scaled.shape[2])
model = create_model(input_shape)
model.summary()

print(x_test_scaled.shape, y_test_scaled.shape)
model.fit(x_train_scaled, y_train_scaled, epochs=50, batch_size=32, validation_split=0.2)

model.save('fastapi/lstm_model_for_fastapi.h5')
model.load_weights('fastapi/lstm_model_for_fastapi.h5')
predictions = model.predict(x_test_scaled)

def evaluate_model(model, x_test_scaled, y_test_scaled, scaler):
    preds = model.predict(x_test_scaled)
    preds_reshaped = preds.reshape(-1, preds.shape[2])
    y_true_reshaped = y_test_scaled.reshape(-1, y_test_scaled.shape[2])
    preds_inv = scaler.inverse_transform(preds_reshaped).reshape(preds.shape)
    y_true_inv = scaler.inverse_transform(y_true_reshaped).reshape(y_test_scaled.shape)
    
    mse = np.mean((preds_inv - y_true_inv) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds_inv - y_true_inv))
    mask = np.abs(y_true_inv) > 1e-4
    mape = np.mean(np.abs((preds_inv - y_true_inv)[mask] / y_true_inv[mask])) * 100
    
    return mse, rmse, mae, mape
print("Evaluating model...")
mse, rmse, mae, mape = evaluate_model(model, x_test_scaled, y_test_scaled, scaler)
print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, MAPE: {mape}")

joblib.dump(scaler, "fastapi/lstm_model_for_fastapi.pkl")

app = FastAPI()

# Load your trained model and scaler (ensure these files exist)
model = tf.keras.models.load_model("fastapi/lstm_model_for_fastapi.h5")
scaler = joblib.load("fastapi/lstm_model_for_fastapi.pkl")

class InputData(BaseModel):
    data: list  # Should be shaped as (input_steps, num_features)

class EvalData(BaseModel):
    x: list  # shape: (batch, input_steps, num_features)
    y: list  # shape: (batch, output_steps, num_features)

@app.post("/predict")
def predict(input_data: InputData):
    x = np.array(input_data.data)
    x_scaled = scaler.transform(x).reshape(1, x.shape[0], x.shape[1])
    pred = model.predict(x_scaled)
    pred_reshaped = pred.reshape(-1, pred.shape[2])
    pred_inv = scaler.inverse_transform(pred_reshaped).reshape(pred.shape)
    return {"prediction": pred_inv.tolist()}

@app.post("/evaluate")
def evaluate(eval_data: EvalData):
    x = np.array(eval_data.x)
    y_true = np.array(eval_data.y)
    # Scale input
    x_scaled = scaler.transform(x.reshape(-1, x.shape[2])).reshape(x.shape)
    # Predict
    preds = model.predict(x_scaled)
    # Inverse transform predictions and y_true
    preds_reshaped = preds.reshape(-1, preds.shape[2])
    y_true_reshaped = y_true.reshape(-1, y_true.shape[2])
    preds_inv = scaler.inverse_transform(preds_reshaped).reshape(preds.shape)
    y_true_inv = scaler.inverse_transform(y_true_reshaped).reshape(y_true.shape)
    # Metrics
    mse = np.mean((preds_inv - y_true_inv) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds_inv - y_true_inv))
    mask = np.abs(y_true_inv) > 1e-4
    mape = np.mean(np.abs((preds_inv - y_true_inv)[mask] / y_true_inv[mask])) * 100
    return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "mape": float(mape)}
