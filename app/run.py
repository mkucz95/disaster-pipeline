import json
import plotly
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from random import randint
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

#import custom classes and function for model
from package.custom_transformer import MessageLengthExtractor, StartingNounExtractor, NumericalExtractor, tokenize
app = Flask(__name__)

# load data
engine = create_engine('sqlite:///app/DisasterResponse.db') #create engine for sql access
df = pd.read_sql_table('disasterResponse', engine) #from table name
categories= df.columns[4:]

# load model
model = joblib.load('app/classifier.pkl')
def random_colors(array):
    return dict(color=['rgb({},{},{})'.format(randint(0,256), randint(0,256), randint(0,256)) for x in array])
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    genre_colors = random_colors(genre_names)
        
    df_requests_1 = df.groupby('request').sum().iloc[1,2:].sort_values(ascending=False)[:10]
    request_type_count = df_requests_1.values
    request_type_index = df_requests_1.index
    request_colors = random_colors(request_type_index)
    
    most_rel_df = pd.DataFrame(df[categories].sum(), columns=['sum']).sort_values('sum', ascending=False).iloc[:18,:]
    most_related = most_rel_df['sum'].values
    most_related_names = most_rel_df.index
    mrn_colors = random_colors(most_related_names)

    least_rel_df = pd.DataFrame(df[categories].sum(), columns=['sum']).sort_values('sum', ascending=False).iloc[18:,:]
    least_related = least_rel_df['sum'].values
    least_related_names = least_rel_df.index
    lrn_colors = random_colors(least_related_names)
    
    df_infrastructure = df[['infrastructure_related', 'transport', 'buildings', 'electricity',
       'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure']].mean()
    infrastructure_counts = df_infrastructure.values
    infrastructure_names = df_infrastructure.index
    infrastructure_colors = random_colors(infrastructure_names)

    df_weather = df[['weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
       'other_weather']].mean()
    weather_mean = df_weather
    weather_names = df_weather.index
    weather_colors = random_colors(weather_names)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=genre_colors
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
                    y=request_type_count,
                    marker=request_colors
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
                    y=most_related,
                    marker=mrn_colors
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
                    y=least_related,
                    marker=lrn_colors
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
                    x=infrastructure_names,
                    y=infrastructure_counts,
                    marker=infrastructure_colors
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
                    y=weather_mean,
                    marker=weather_colors
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
    classification_results = dict(zip(categories, classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

#def main():
 #   app.run(host='0.0.0.0', port=3001, debug=True)

#if __name__ == '__main__':
    #main()