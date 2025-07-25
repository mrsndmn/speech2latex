import datasets
import pandas as pd

if __name__ == "__main__":

    dataset_path = "./MathSpeech_whisper_transcribed"
    dataset = datasets.load_from_disk(dataset_path)

    df = pd.DataFrame({
        'whisper_text': dataset['whisper_text'],
        'transcription': dataset['transcription'],
        'latex': dataset['LaTeX'],
    })

    df.to_csv('./MathSpeech_whisper_transcribed.csv', index=False)

    print(df.head())
