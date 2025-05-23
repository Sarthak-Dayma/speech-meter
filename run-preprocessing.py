from src.preprocessing import preprocess_transcripts_chunked

if __name__ == "__main__":
    input_csv = "data/raw/ted_talks_en.csv"
    output_csv = "data/processed/ted_chunks_with_features.csv"
    
    preprocess_transcripts_chunked(input_csv, output_csv, max_talks=500, chunk_size=3)
