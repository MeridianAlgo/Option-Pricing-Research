import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# For explainability
try:
    import shap
except ImportError:
    shap = None

class OptionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x):
        return self.net(x)

class OptionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class OptionTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model)
        out = self.transformer(x)
        out = out[-1]  # last time step
        return self.fc(out)

def fit_ml_model(X, y, epochs=100, lr=1e-3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    model = OptionMLP(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).numpy().flatten()
        mse = mean_squared_error(y_test, y_pred)
        print(f"Test MSE: {mse:.4f}")
    return model, scaler

def predict_ml_model(model, scaler, X):
    X_scaled = scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X_t).numpy().flatten()
    return y_pred

def fit_lstm_model(X_seq, y, epochs=100, lr=1e-3):
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    model = OptionLSTM(X_seq.shape[2])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"LSTM Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).numpy().flatten()
        mse = mean_squared_error(y_test, y_pred)
        print(f"LSTM Test MSE: {mse:.4f}")
    return model

def predict_lstm_model(model, X_seq):
    X_t = torch.tensor(X_seq, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X_t).numpy().flatten()
    return y_pred

def fit_transformer_model(X_seq, y, epochs=100, lr=1e-3):
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    model = OptionTransformer(X_seq.shape[2])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Transformer Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).numpy().flatten()
        mse = mean_squared_error(y_test, y_pred)
        print(f"Transformer Test MSE: {mse:.4f}")
    return model

def predict_transformer_model(model, X_seq):
    X_t = torch.tensor(X_seq, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X_t).numpy().flatten()
    return y_pred

def explain_model_with_shap(model, X, scaler=None):
    if shap is None:
        print("SHAP not installed. Skipping explainability.")
        return
    if scaler:
        X = scaler.transform(X)
    explainer = shap.DeepExplainer(model, torch.tensor(X, dtype=torch.float32))
    shap_values = explainer.shap_values(torch.tensor(X, dtype=torch.float32))
    shap.summary_plot(shap_values, X)

def auto_retrain_ml_model(X, y, model_type='MLP', epochs=100, lr=1e-3):
    """
    Retrain ML model when new data arrives. Supports MLP, LSTM, Transformer.
    """
    if model_type == 'MLP':
        return fit_ml_model(X, y, epochs=epochs, lr=lr)
    elif model_type == 'LSTM':
        return fit_lstm_model(X, y, epochs=epochs, lr=lr)
    elif model_type == 'Transformer':
        return fit_transformer_model(X, y, epochs=epochs, lr=lr)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def blend_models(X, models, scalers=None, weights=None):
    """
    Ensemble predictions from multiple models (MLP, LSTM, Transformer, Black-Scholes).
    X: input features (or sequence)
    models: list of (model, type) tuples
    scalers: list of scalers (optional)
    weights: list of weights (optional)
    Returns: blended prediction
    """
    preds = []
    for i, (model, mtype) in enumerate(models):
        scaler = scalers[i] if scalers else None
        if mtype == 'MLP':
            pred = predict_ml_model(model, scaler, X)
        elif mtype == 'LSTM':
            pred = predict_lstm_model(model, X)
        elif mtype == 'Transformer':
            pred = predict_transformer_model(model, X)
        elif mtype == 'Black-Scholes':
            # X: [S, K, T, r, sigma, type]
            pred = np.array([black_scholes_row(row) for row in X])
        else:
            raise ValueError(f"Unknown model type: {mtype}")
        preds.append(pred)
    preds = np.array(preds)
    if weights is None:
        weights = np.ones(len(models)) / len(models)
    blended = np.average(preds, axis=0, weights=weights)
    return blended

def black_scholes_row(row):
    from option_pricing import OptionPricing
    S, K, T, r, sigma, otype = row[:6]
    op = OptionPricing(S, K, T, r, sigma, otype)
    return op.black_scholes()[0]
