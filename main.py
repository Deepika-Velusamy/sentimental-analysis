"'importing the required libraries for the program'"
import pandas as pd
import numpy as np
import requests

import sklearn
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


"'LOADING the data'"
np.random.seed(40)
def read_data(pos, neg):
    pos = pd.read_csv(pos, sep="\n", header=None, names=['review'])
    pos['positive']=1
    neg = pd.read_csv(neg, sep="\n", header=None, names=['review'])
    neg['positive']=0
    combined_df = pos.append(neg)
    combined_df = shuffle(combined_df, random_state=42)
    return(combined_df)

train = read_data(pos="dataset/imdb_train_pos.txt",
                  neg="dataset/imdb_train_neg.txt")

dev = read_data(pos="dataset/imdb_dev_pos.txt",
                neg="dataset/imdb_dev_neg.txt")

test = read_data(pos="dataset/imdb_test_pos.txt",
                 neg="dataset/imdb_test_neg.txt")


"'FEATURE 1 - tf-idf'"
stopwords=set(nltk.corpus.stopwords.words('english'))
stopwords.add(";")
stopwords.add("/")
stopwords.add(">")
stopwords.add("<")
stopwords.add("br")
stopwords.add("(")
stopwords.add(")")
stopwords.add("''")
stopwords.add("&")
stopwords.add(".")
stopwords.add(",")
stopwords.add("--")
stopwords.add("``")
stopwords.add("#")
stopwords.add("@")
stopwords.add(":")
stopwords.add("'s")
stopwords.add("’")
stopwords.add("...")
stopwords.add("n't")
stopwords.add("'re")
stopwords.add("'")
stopwords.add("-")

"'creating classes for perforing fit and transform of the models selected'"
class selectReview(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return(X['review'])

feature_1_vocab = Pipeline([('select_review', selectReview()),
    ('count', CountVectorizer(stop_words=stopwords, max_features=500)),
    ('tfidf', TfidfTransformer())])        

"'FEATURE 2 - tf-idf bigram'"
feature_2_vocab = Pipeline([
    ('select_review', selectReview()),
    ('count', CountVectorizer(stop_words=stopwords, max_features=500, ngram_range=(2,2))),
    ('tfidf', TfidfTransformer())])

"'FEATURE 3 Sentimental Analysis'"
vader = SentimentIntensityAnalyzer()

"'Creating coustum transfermation class and pipeline'"
class getSentiment(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        features_array=[]
        for index, row in X.iterrows():
            pos = vader.polarity_scores(row['review'])['pos']
            neu = vader.polarity_scores(row['review'])['neu']
            neg = vader.polarity_scores(row['review'])['neg']
            features_array.append([pos, neu, neg])
        return(np.asarray(features_array))


feature_3_sentiment = Pipeline([('get_sentiment', getSentiment())])



"'Feature_Engineering'"

feature_engineering = FeatureUnion(transformer_list=[
    ("feature_1_vocab", feature_1_vocab),
    ("feature_2_vocab", feature_2_vocab),
    ("feature_3_sentiment", feature_3_sentiment)])


X_train = feature_engineering.fit_transform(train)

"'Evoluating the models in development set'"
y_train = np.asarray(train['positive'])
y_dev = np.asarray(dev['positive'])
X_dev = feature_engineering.transform(dev)

"'SVM Linear'"

svm_clf = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("svm_clf", sklearn.svm.LinearSVC(loss='hinge'))])
svm_clf.fit(X_train, y_train)
svm_clf_pred = svm_clf.predict(X_dev)
print("SVM Linear classification_report")
print(classification_report(y_dev, svm_clf_pred))
print("----------------------------------------------------")

"'RBF'"
rbf_svm_clf = Pipeline([
    ("scaler", StandardScaler(with_mean=False)),
    ("svm_clf", sklearn.svm.SVC(kernel="rbf"))])
rbf_svm_clf.fit(X_train, y_train)
rbf_svm_clf_pred = rbf_svm_clf.predict(X_dev)
print("RBF classification_report")
print(classification_report(y_dev, rbf_svm_clf_pred))
print("----------------------------------------------------")

"'Decission Tree'"
tree_clf = DecisionTreeClassifier(max_depth=3)
tree_clf.fit(X_train, y_train)
tree_pred = tree_clf.predict(X_dev)
print("Decission Tree classification_report")
print(classification_report(y_dev, tree_pred))
print("----------------------------------------------------")

"'Hypertuning using gride search'"

param_grid = {'svm_clf__C': [0.01, 0.1, 1.0]}
grid_search = GridSearchCV(rbf_svm_clf, param_grid, scoring='f1_weighted')
for i in [10,100,500,1000]:
    selector = SelectKBest(chi2, k=i)
    X_train_reduced = selector.fit_transform(X_train, np.asarray(train['positive']))
    grid_search.fit(X_train_reduced, y_train)
    print('\nNo of features: '+str(i))
    print(grid_search.best_params_)
    print(round(grid_search.best_score_, 3))

"'Evaluating test dataset'"
selector = SelectKBest(chi2, k=1000)
X_train_reduced = selector.fit_transform(X_train, y_train)
y_test = np.asarray(test['positive'])
X_test = feature_engineering.transform(test)
X_test_reduced = selector.transform(X_test)
rbf_svm_clf.fit(X_train_reduced, y_train)
rbf_svm_clf_pred = rbf_svm_clf.predict(X_test_reduced)

print('F-measure')
print(round(f1_score(y_test, rbf_svm_clf_pred, average='weighted'), 3))
print('Precision')
print(round(precision_score(y_test, rbf_svm_clf_pred), 3))
print('Recall')
print(round(recall_score(y_test, rbf_svm_clf_pred), 3))
print('\nAccuracy')
print(round(accuracy_score(y_test, rbf_svm_clf_pred), 3))












