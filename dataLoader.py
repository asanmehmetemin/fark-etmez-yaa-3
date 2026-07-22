import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    @staticmethod
    def load_imdb():
        data = pd.read_csv('imdb.csv')
        df = data.dropna(subset=['review', 'sentiment'])
        df['Label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        return df['review'].values, df['Label'].values
    
    @staticmethod
    def load_tweets():
        df= pd.read_csv('tweets.csv')
        df.dropna(subset=['text', 'sentiment'], inplace=True)
        tweet_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
        df['Label'] = df['sentiment'].map(tweet_mapping)
        return df['text'].values, df['Label'].values
    
    @staticmethod
    def split_data(x, y):
        x_temp,x_test,y_temp,y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        x_train,x_val,y_train,y_val = train_test_split(x_temp, y_temp,test_size=0.2, random_state=42)
        return x_train, x_val, x_test, y_train, y_val, y_test
