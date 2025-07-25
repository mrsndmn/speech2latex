import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('./latex_in_context_tts_v2_full.csv')
    
    df['pronunciation'] = df['pronunciation'].fillna( df['text'] )
    assert df['pronunciation'].isna().sum() == 0

    # filter out whisper hallucinations
    df = df[df['pronunciation'].apply(len) * 1.5 > df['whisper_text'].apply(len)]

    # filter out too long sequences
    df = df[ df['whisper_text'].apply(len) < 300 ]

    train = df.sample(frac=0.95, random_state=0)
    test = df.drop(train.index)
    print("train", len(train), "test", len(test), "df", len(df))
    assert len(train) + len(test) == len(df)
    
    train.to_csv('./latex_in_context_tts_v2_train.csv')
    test.to_csv('./latex_in_context_tts_v2_test.csv')

    breakpoint()
