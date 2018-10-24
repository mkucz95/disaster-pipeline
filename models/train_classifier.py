import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import nltk
from sklearn.externals import joblib

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

class MessageLengthExtractor(BaseEstimator, TransformerMixin):
    def message_length(self, text):
        '''
        Returns the number of characters in text
        '''
        return len(text)
    
    def fit(self, x, y=None):
        '''
         Overriding function from baseclass, fits the object
        '''
        return self
    
    def transform(self, X):
        '''
         Overriding function from baseclass, transforms the object
        '''
        X_msg_len = pd.Series(X).apply(self.message_length)
        return (pd.DataFrame(X_msg_len))

class StartingNounExtractor(BaseEstimator, TransformerMixin):
    def starting_noun(self, text):
        '''
        Is there a sentence that starts with a Noun
        '''
        sentences= nltk.sent_tokenize(text)
        for sentence in sentences:
            parts_of_speech_tags = nltk.pos_tag(tokenize(sentence))
            word_1, tag_1 = parts_of_speech_tags[0]
            if(tag_1[:2]=='NN'):
                return True
        return False
    
    def fit(self, X, y=None):
        '''
         Overriding function from baseclass, fits the object
        '''
        return self
    
    def transform(self, X):
        '''
         Overriding function from baseclass, transforms the object
        '''
        X_tagged = pd.Series(X).apply(self.starting_noun)
        return(pd.DataFrame(X_tagged))

class NumericalExtractor(BaseEstimator, TransformerMixin):
    def has_numerical(self, text):
        '''
         returns whether the text contains a number in it
        '''
        pos_tags = nltk.pos_tag(tokenize(text))
        for word, tag in pos_tags:
            if(tag[:3]=='NUM'): return True
        return False
    
    def fit(self, X, y=None):
        '''
         Overriding function from baseclass, fits the object
        '''
        return self
    
    def transform(self, X):
        '''
         Overriding function from baseclass, transforms the object
        '''
        X_tagged = pd.Series(X).apply(self.has_numerical)
        return(pd.DataFrame(X_tagged))

def load_data(database_filepath):
    '''
    Load data from SQL Database into pandas DataFrame object
    Input: filepath to SQL database location
    Output: X values, and Y values, as well as the column names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM disasterResponse", engine)
    X= df.message.values
    Y = df.iloc[:,4:]
    return X, Y, df.columns[4:]

def tokenize(text):
    '''
    Returns the tokenization of text. Splits the text into individual words and
    then transforms each word into its root. The words are also cleaned of any punctuation
    
    Output:array of cleaned and tokenized words
    '''
    tokens = word_tokenize(text) #split each message into individual words
    lemmatizer = WordNetLemmatizer()
    clean_tokens=[]
    for token in tokens:
        #clean each token from whitespace and punctuation, and conver to
        #root of word ie walking->walk
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    return clean_tokens

def build_model():
    '''
    Builds ML pipeline. 1. get the matrix of counts for each word in messages
    2. do tfidf transformation which shows occurence in realtion to number of documents
    3. perform adaboostclassifier on multioutput classifier
    Contain the pipeline object within gridearch, so that the pipeline is optimized when being fit.
    Output: GridSearch object
    '''
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('textpipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('mesg_len', MessageLengthExtractor()),
            ('noun_start', StartingNounExtractor()),
            ('contains_num', NumericalExtractor())
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    parameters = {
        'features__textpipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__learning_rate': [0.1, 0.5, 1]
    }
return GridSearchCV(pipeline, param_grid=parameters)

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluation of model for each classification cateogry
    Input: model object, X_test, Y_test, names of categories
    Output: prints the classification report for each model and category
    '''
    predictions = pd.DataFrame(model.predict(X_test), columns = y_test.columns)
    for col in category_names:
        print(col)
        print(classification_report(Y_test[col], predictions[col]))
        
def save_model(model, model_filepath):
    '''
    Save a model to certain filepath
    Input: model object, filepath
    '''
    joblib.dump(model, model_filepath);

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()