import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sentence_transformers import SentenceTransformer

def train_with_sentence_embeddings(input_csv):
    df = pd.read_csv(input_csv)
    X = df['chunk_text'].tolist()
    y = df['completion_percent']

    # Load pre-trained model
    model_embed = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed texts
    X_embedded = model_embed.encode(X, show_progress_bar=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_embedded, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name} on sentence embeddings...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        results[name] = {'MSE': mse, 'MAE': mae}
        print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}")

    return models, results
