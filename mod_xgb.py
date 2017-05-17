from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import operator
from sklearn.cross_validation import train_test_split
from collections import Counter
from nltk.corpus import stopwords
from cleaner import myclean
from sklearn.pipeline import Pipeline
from collections import Counter
import pickle

class textCleaner(BaseEstimator,TransformerMixin):
    """
    input: raw df
    output: same df with text cleaned
    """
    def __init__(self):
        pass

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        X.question1 = X.question1.apply(lambda x: myclean(str(x)))
        X.question2 = X.question2.apply(lambda x: myclean(str(x)))
        return X

# https://www.kaggle.com/dasolmar/xgb-with-whq-jaccard/code/code
class featureGenerator(BaseEstimator,TransformerMixin):
    """
    input: a cleaned df
    output: data for xgb model
    """
    def __init__(self,df_train,df_test,eps=10000,add_5w = True,add_magic = True):
        self.eps = eps
        self.add_5w = add_5w
        self.add_magic = add_magic
        self.df_train = df_train # for calculating static mapper
        self.df_test = df_test # for calculating static mapper
        #weights
        train_qs = pd.Series(self.df_train['question1'].tolist() + self.df_train['question2'].tolist()).astype(str)
        words = (" ".join(train_qs)).lower().split()
        counts = Counter(words)
        self.weights = {word: self.get_weight(count) for word, count in counts.items()}
        #magic feature
        all_question = np.concatenate((self.df_train.question1.values,self.df_train.question2.values,
                                      self.df_test.question1.values,self.df_test.question2.values))
        self.hash_mapper = {v:i for i, v in enumerate(set(all_question))}
        # how many times each question appeared in the whole curpus
        self.count_mapper = Counter(all_question)

    def get_weight(self,count, min_count=2):
        return 0 if count < min_count else 1 / (count + self.eps)

    def word_shares(self,row):
        stops = set(stopwords.words("english"))

        q1_list = str(row['question1']).split()
        q1 = set(q1_list)
        q1words = q1.difference(stops)
        if len(q1words) == 0:
            return 'nan:'*11 + 'nan'

        q2_list = str(row['question2']).split()
        q2 = set(q2_list)
        q2words = q2.difference(stops)
        if len(q2words) == 0:
            return 'nan:'*11 + 'nan'

        # % of words at the same location
        words_hamming = sum(1 for i in zip(q1_list, q2_list) if i[0] == i[1]) \
                        / max(len(q1_list), len(q2_list))

        q1stops = q1.intersection(stops)
        q2stops = q2.intersection(stops)

        q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])
        q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])

        shared_2gram = q1_2gram.intersection(q2_2gram)

        # num of shared words
        shared_words = q1words.intersection(q2words)
        shared_weights = [self.weights.get(w, 0) for w in shared_words]
        q1_weights = [self.weights.get(w, 0) for w in q1words]
        q2_weights = [self.weights.get(w, 0) for w in q2words]
        total_weights = q1_weights + q2_weights

        R1 = np.sum(shared_weights) / np.sum(total_weights)  # tfidf share
        R2 = len(shared_words) / (len(q1words) + len(q2words) - len(shared_words))  # count share
        R31 = len(q1stops) / len(q1words)  # % of stops in q1
        R32 = len(q2stops) / len(q2words)  # % of stops in q2
        Rcosine_denominator = (np.sqrt(np.dot(q1_weights, q1_weights)) * np.sqrt(np.dot(q2_weights, q2_weights)))
        Rcosine = np.dot(shared_weights, shared_weights) / Rcosine_denominator
        if len(q1_2gram) + len(q2_2gram) == 0:
            R2gram = 0
        else:
            R2gram = len(shared_2gram) / (len(q1_2gram) + len(q2_2gram))

        q1_hash = self.hash_mapper[row['question1']]
        q2_hash = self.hash_mapper[row['question2']]

        q1_freq = self.count_mapper[row['question1']]
        q2_freq = self.count_mapper[row['question2']]

        rslt = '{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words),
                                                            R31, R32, R2gram, Rcosine, words_hamming,
                                                            q1_hash,q2_hash,q1_freq,q2_freq)

        return rslt

    def add_word_count(self, x, df, word):
        x['q1_' + word] = df['question1'].apply(lambda x: (word in str(x).lower()) * 1)
        x['q2_' + word] = df['question2'].apply(lambda x: (word in str(x).lower()) * 1)
        x[word + '_both'] = x['q1_' + word] * x['q2_' + word]

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        df = X
        df['word_shares'] = df.apply(self.word_shares, axis=1, raw=True)
        x = pd.DataFrame()

        x['word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[0]))
        x['word_match_2root'] = np.sqrt(x['word_match'])
        x['tfidf_word_match'] = df['word_shares'].apply(lambda x: float(x.split(':')[1]))
        x['shared_count'] = df['word_shares'].apply(lambda x: float(x.split(':')[2]))

        x['stops1_ratio'] = df['word_shares'].apply(lambda x: float(x.split(':')[3]))
        x['stops2_ratio'] = df['word_shares'].apply(lambda x: float(x.split(':')[4]))
        x['shared_2gram'] = df['word_shares'].apply(lambda x: float(x.split(':')[5]))
        x['cosine'] = df['word_shares'].apply(lambda x: float(x.split(':')[6]))
        x['words_hamming'] = df['word_shares'].apply(lambda x: float(x.split(':')[7]))


        x['q1_hash_index'] = df['word_shares'].apply(lambda x: float(x.split(':')[8]))
        x['q2_hash_index'] = df['word_shares'].apply(lambda x: float(x.split(':')[9]))
        x['q1_freq_count'] = df['word_shares'].apply(lambda x: float(x.split(':')[10]))
        x['q2_freq_count'] = df['word_shares'].apply(lambda x: float(x.split(':')[11]))


        x['diff_stops_r'] = x['stops1_ratio'] - x['stops2_ratio']

        x['len_q1'] = df['question1'].apply(lambda x: len(str(x)))
        x['len_q2'] = df['question2'].apply(lambda x: len(str(x)))
        x['diff_len'] = x['len_q1'] - x['len_q2']

        x['caps_count_q1'] = df['question1'].apply(lambda x: sum(1 for i in str(x) if i.isupper()))
        x['caps_count_q2'] = df['question2'].apply(lambda x: sum(1 for i in str(x) if i.isupper()))
        x['diff_caps'] = x['caps_count_q1'] - x['caps_count_q2']

        x['len_char_q1'] = df['question1'].apply(lambda x: len(str(x).replace(' ', '')))
        x['len_char_q2'] = df['question2'].apply(lambda x: len(str(x).replace(' ', '')))
        x['diff_len_char'] = x['len_char_q1'] - x['len_char_q2']

        x['len_word_q1'] = df['question1'].apply(lambda x: len(str(x).split()))
        x['len_word_q2'] = df['question2'].apply(lambda x: len(str(x).split()))
        x['diff_len_word'] = x['len_word_q1'] - x['len_word_q2']

        x['avg_world_len1'] = x['len_char_q1'] / x['len_word_q1']
        x['avg_world_len2'] = x['len_char_q2'] / x['len_word_q2']
        x['diff_avg_word'] = x['avg_world_len1'] - x['avg_world_len2']

        x['exactly_same'] = (df['question1'] == df['question2']).astype(int)
        x['duplicated'] = df.duplicated(['question1', 'question2']).astype(int)

        if self.add_5w:
            self.add_word_count(x, df, 'how')
            self.add_word_count(x, df, 'what')
            self.add_word_count(x, df, 'which')
            self.add_word_count(x, df, 'who')
            self.add_word_count(x, df, 'where')
            self.add_word_count(x, df, 'when')
            self.add_word_count(x, df, 'why')

        return x

class moreFeature(BaseEstimator,TransformerMixin):
    """
    input: raw df
    output: same df with text cleaned
    """
    def __init__(self,train,test):
        self.train = train
        self.test = test

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        X.question1 = X.question1.apply(lambda x: myclean(str(x)))
        X.question2 = X.question2.apply(lambda x: myclean(str(x)))
        return X



def train_xgb(X, y, params):
    """
    input: processed data
    output: xgb model
    """
    print("train XGB ")
    x, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)

    dtrain = xgb.DMatrix(x, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    return xgb.train(params, dtrain,num_boost_round=10000, evals=watchlist, early_stopping_rounds=10)


def predict_xgb(clf, X_test):
    return clf.predict(xgb.DMatrix(X_test))


if __name__ == '__main__':
    train_raw = pd.read_csv('./input/train.csv')
    test_raw = pd.read_csv('./input/test.csv')

    train_cleaned = textCleaner().fit_transform(train_raw)
    test_cleaned = textCleaner().fit_transform(test_raw)

    print('finish cleaning')

    x_train = featureGenerator(train_cleaned,test_cleaned).fit_transform(train_cleaned)
    x_test = featureGenerator(train_cleaned, test_cleaned).fit_transform(test_cleaned)
    y_train = train_cleaned.is_duplicate




    pickle.dump(x_train,open('./output/x_train.pkl','wb'))
    pickle.dump(x_test, open('./output/x_test.pkl', 'wb'))
    pickle.dump(y_train, open('./output/y_train.pkl', 'wb'))

    # train xgb
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.11
    params['max_depth'] = 5
    params['silent'] = 1
    params['seed'] = 123

    clf = train_xgb(x_train,y_train,params)
    preds = predict_xgb(clf,x_test)

    print('make submit...')
    sub = pd.DataFrame({'test_id':test_raw.test_id,'is_duplicate':preds})
    sub.to_csv('./output/sub_{}.csv'.format(datetime.datetime.now().strftime('%m%d_%H%M')),index=False)



