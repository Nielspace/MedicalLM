from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


class embeddings:

    """
        returns: X_train, X_test, y_train, y_test, vectorizer, feature_names, pca, labels, category_list
    """
    @staticmethod
    def embeddings_gen(df, 
                X_features,
                y_features, 
                n_components=0.95,
                test_size=0.2,
                max_features=1000, 
                random_state=23):

        vectorizer = TfidfVectorizer(analyzer='word', stop_words='english',ngram_range=(1,3), 
                                    max_df=0.75, use_idf=True, smooth_idf=True, max_features=max_features)

        X_ = df[X_features]
        y_ = df[y_features]

        tfIdfMat  = vectorizer.fit_transform(X_.tolist())
        feature_names = sorted(vectorizer.get_feature_names_out())

        pca = PCA(n_components=n_components, random_state=random_state)
        tfIdfMat_reduced = pca.fit_transform(tfIdfMat.toarray())
        labels = y_.tolist()
        category_list = y_.unique()
        X_train, X_test, y_train, y_test = train_test_split(tfIdfMat_reduced, labels, 
                                                            test_size=test_size, 
                                                            stratify=labels,
                                                            random_state=random_state)
        
        return (X_train, X_test, y_train, y_test, vectorizer, feature_names, pca, labels, category_list)
        
