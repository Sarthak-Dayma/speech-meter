import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_and_evaluate_all_models(processed_csv_path):
    # Load data
    df = pd.read_csv(processed_csv_path)
    y = df['completion_percent'].values
    numeric_features = df[[
        'num_sentences', 'num_words', 'avg_word_len', 'rel_position',
        'flesch_reading_ease', 'flesch_kincaid_grade', 'sentiment',
        'cum_unique_words_count', 'cum_total_words_count', 'cum_unique_ratio'
    ]].values

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(df['chunk_text'])

    # Combine TF-IDF and numeric features
    X_combined = np.hstack([X_tfidf.toarray(), numeric_features])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42
    )

    print("\nTraining Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    preds_lr = lr.predict(X_test)
    print(f"Linear Regression - MSE: {mean_squared_error(y_test, preds_lr):.4f}, MAE: {mean_absolute_error(y_test, preds_lr):.4f}")

    print("\nTuning Gradient Boosting...")
    gb_params = {
        'n_estimators': [100, 150],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1]
    }
    gb = GradientBoostingRegressor(random_state=42)
    gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    gb_grid.fit(X_train, y_train)
    best_gb = gb_grid.best_estimator_
    preds_gb = best_gb.predict(X_test)
    print("Best Gradient Boosting Params:", gb_grid.best_params_)
    print(f"Gradient Boosting - MSE: {mean_squared_error(y_test, preds_gb):.4f}, MAE: {mean_absolute_error(y_test, preds_gb):.4f}")

    print("\nTuning Random Forest...")
    rf_params = {
        'n_estimators': [100, 150],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    preds_rf = best_rf.predict(X_test)
    print("Best Random Forest Params:", rf_grid.best_params_)
    print(f"Random Forest - MSE: {mean_squared_error(y_test, preds_rf):.4f}, MAE: {mean_absolute_error(y_test, preds_rf):.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python modeling_tfidf.py path_to_processed_csv")
    else:
        train_and_evaluate_all_models(sys.argv[1])
