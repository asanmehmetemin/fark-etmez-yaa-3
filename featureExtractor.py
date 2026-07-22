from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    def __init__(self, max_features = 5000):
        self.max_features = max_features
        self.vectorizer = None

    def fit_transform_train(self, x_train):
        self.vectorizer = TfidfVectorizer(max_features=self.max_features)
        x_train_tfidf = self.vectorizer.fit_transform(x_train)

        return x_train_tfidf.toarray()
    
    def transform_unseen(self, x_val_test):
        x_val_test_tfidf = self.vectorizer.transform(x_val_test)

        return x_val_test_tfidf.toarray()