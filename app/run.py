import json
import plotly
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

from flask import Flask, render_template, request, jsonify
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
        sentences= sent_tokenize(text)
        for sentence in sentences:
            parts_of_speech_tags = pos_tag(tokenize(sentence))
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

def tokenize(text):
    '''
    returns the tokenization of the text, necessary for the custom classification of messages
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///DisasterResponse.db') #create engine for sql access
df = pd.read_sql_table('disasterResponse', engine) #from table name

# load model
model = joblib.load("classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    message_type = ['related', 'request', 'offer']
    message_type_count = [df[x].value_counts().values for x in message_type]
    message_type_index = [df[x].value_counts().index for x in message_type]
    
    df_requests_1 = df.groupby('request').sum().iloc[1,2:].sort_values(ascending=False)[:10]
    request_type_count = df_requests_1.values
    request_type_index = df_requests_1.index
    
    most_rel_df = pd.DataFrame(df.iloc[:,4:].sum(), columns=['sum']).sort_values('sum', ascending=False).iloc[:18,:]
    most_related = most_rel_df['sum'].values
    most_related_names = most_rel_df.index
    least_rel_df = pd.DataFrame(df.iloc[:,4:].sum(), columns=['sum']).sort_values('sum', ascending=False).iloc[18:,:]
    least_related = least_rel_df['sum'].values
    least_related_names = least_rel_df.index
    
    df_infrastructure = df[['infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure']].mean()
    infrastructure_counts = df_infrastructure.values
    infrastructure_names = df_infrastructure.index
    
    df_weather = df[['weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather']].mean()
    weather_mean = df_weather
    weather_names = df_weather.index
    
    # create visuals
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
                    x=request_type_index,
                    y=request_type_count
                )
            ],

            'layout': {
                'title': 'Most Common Request Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Type of Request"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=most_related_names,
                    y=most_related
                )
                    ],

            'layout': {
                'title': 'Most Common Messages',
                'yaxis': {
                    'title': "Count",
                    'showspikes':"true"
                },
                'xaxis': {
                    'title': "Type",
                },
            }
        },
        {
            'data': [
                Bar(
                    x=least_related_names,
                    y=least_related
                )
                    ],
            'layout': {
                'title': 'Least Common Messages',
                'yaxis': {
                    'title': "Count",
                    'showspikes':"true"
                },
                'xaxis': {
                    'title': "Type",
                },
            }
        },
        {
            'data': [
                Bar(
                    x=most_related_names,
                    y=most_related
                )
                    ],

            'layout': {
                'title': 'Most Common Messages',
                'yaxis': {
                    'title': "Count",
                    'showspikes':"true"
                },
                'xaxis': {
                    'title': "Type",
                },
            }
        },
        {
            'data': [
                Bar(
                    x=infrastructure_names,
                    y=infrastructure_counts
                )
                    ],

            'layout': {
                'title': 'Infrastructure Related Messages',
                'yaxis': {
                    'title': "% of all messages",
                    'showspikes':"true",
                    'tickformat': ',.2%',
                     'range': '[0,0.5]'
                },
                'xaxis': {
                    'title': "Infrastructure Type",
                },
            }
        },
        {
            'data': [
                Bar(
                    x=weather_names,
                    y=weather_mean
                )
                    ],
            'layout': {
                'title': 'Weather Related Messages',
                'yaxis': {
                    'title': "% of all messages",
                    'showspikes':"true",
                    'tickformat': ',.2%',
                     'range': '[0,0.5]'
                },
                'xaxis': {
                    'title': "Message Type",
                },
            }
        }, 
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

#def main():
 #   app.run(host='0.0.0.0', port=3001, debug=True)

#if __name__ == '__main__':
#    main()