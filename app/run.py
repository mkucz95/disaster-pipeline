import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

class MessageLengthExtractor(BaseEstimator, TransformerMixin):
    def message_length(self, text):
        '''
        Returns the number of characters in text
        '''
        return len(text)
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
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
        return self
    
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_noun)
        return(pd.DataFrame(X_tagged))

class NumericalExtractor(BaseEstimator, TransformerMixin):
    def has_numerical(self, text):
        pos_tags = nltk.pos_tag(tokenize(text))
        for word, tag in pos_tags:
            if(tag[:3]=='NUM'): return True
        return False
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.has_numerical)
        return(pd.DataFrame(X_tagged))

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db') #create engine for sql access
df = pd.read_sql_table('disasterResponse', engine) #from table name

# load model
model = joblib.load("./models/ada_classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()