import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin

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