from src.modelingst import train_with_sentence_embeddings

if __name__ == "__main__":
    input_csv = "data/processed/ted_chunks_with_completion.csv"
    models, results = train_with_sentence_embeddings(input_csv)

    print("\nSummary of model performances with embeddings:")
    for model_name, metrics in results.items():
        print(f"{model_name}: MSE={metrics['MSE']:.4f}, MAE={metrics['MAE']:.4f}")
