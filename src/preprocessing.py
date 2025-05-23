import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from src.utils import clean_text
import textstat
from textblob import TextBlob

nltk.download('punkt')

def preprocess_transcripts_chunked(input_csv, output_csv, max_talks=100, chunk_size=3):
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=['transcript'])
    df = df.head(max_talks)

    processed_data = []

    for _, row in df.iterrows():
        talk_id = str(row['talk_id'])  # Ensure talk_id is a string
        transcript = row['transcript']
        sentences = sent_tokenize(transcript)
        total_sentences = len(sentences)

        for i in range(0, total_sentences, chunk_size):
            chunk_sentences = sentences[i:i + chunk_size]
            raw_chunk_text = " ".join(chunk_sentences)
            cleaned_chunk_text = clean_text(raw_chunk_text)

            completion = round(((min(i + chunk_size, total_sentences)) / total_sentences) * 100, 2)
            processed_data.append({
                'talk_id': talk_id,
                'chunk_id': i // chunk_size + 1,
                'chunk_text': cleaned_chunk_text,
                'completion_percent': completion
            })

    processed_df = pd.DataFrame(processed_data)
    processed_df = extract_additional_features(processed_df)
    processed_df.to_csv(output_csv, index=False)
    print(f"Preprocessing chunked transcripts complete. Saved to {output_csv}")


def extract_additional_features(df):
    features = []
    talk_word_tracker = {}

    for idx, row in df.iterrows():
        talk_id = str(row['talk_id'])  # Ensure it's a string
        chunk_text = row['chunk_text']
        total_chunks = df[df['talk_id'] == talk_id].shape[0]
        chunk_id = row['chunk_id']

        sentences = sent_tokenize(chunk_text)
        words = word_tokenize(chunk_text)

        num_sentences = len(sentences)
        num_words = len(words)
        avg_word_len = sum(len(w) for w in words) / num_words if num_words else 0
        rel_position = chunk_id / total_chunks if total_chunks else 0

        flesch_reading_ease = textstat.flesch_reading_ease(chunk_text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(chunk_text)
        sentiment = TextBlob(chunk_text).sentiment.polarity

        # Initialize cumulative tracking
        if talk_id not in talk_word_tracker:
            talk_word_tracker[talk_id] = set()
            talk_word_tracker[talk_id + "_cum_words"] = 0

        current_unique_words = set(w.lower() for w in words if w.isalpha())
        talk_word_tracker[talk_id].update(current_unique_words)
        talk_word_tracker[talk_id + "_cum_words"] += num_words

        cum_unique_words_count = len(talk_word_tracker[talk_id])
        cum_total_words_count = talk_word_tracker[talk_id + "_cum_words"]
        cum_unique_ratio = cum_unique_words_count / cum_total_words_count if cum_total_words_count else 0

        features.append([
            num_sentences,
            num_words,
            avg_word_len,
            rel_position,
            flesch_reading_ease,
            flesch_kincaid_grade,
            sentiment,
            cum_unique_words_count,
            cum_total_words_count,
            cum_unique_ratio
        ])

    feature_df = pd.DataFrame(features, columns=[
        'num_sentences', 'num_words', 'avg_word_len', 'rel_position',
        'flesch_reading_ease', 'flesch_kincaid_grade', 'sentiment',
        'cum_unique_words_count', 'cum_total_words_count', 'cum_unique_ratio'
    ])
    return pd.concat([df.reset_index(drop=True), feature_df], axis=1)
