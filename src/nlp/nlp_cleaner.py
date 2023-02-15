import nltk
nltk.download('stopwords')

import re
import string

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob
import pandas as pd


def clean_word(w):
    text = str(TextBlob(re.sub(fr'[{string.punctuation}{string.digits}]',' ', w)).correct())
    return re.sub(' +', ' ', text)

counter = 0
#Stem and make lower case
def stemSentence(sentence, clean_words=True):
    global counter
    print(counter)
    counter+=1
    stemmer = nltk.SnowballStemmer("english", ignore_stopwords=True)
    token_words = nltk.word_tokenize(str(sentence))
    if clean_words:
        token_words = [clean_word(t) for t in token_words]
    stem_sentence = [stemmer.stem(word) for word in token_words]
    return ' '.join(stem_sentence)



def clean_descriptions(file_name):
    df=pd.read_csv( f'data/{file_name}.csv', encoding='latin1' ).reset_index()
    text1=df['abstract']
    text2=pd.Series( [stemSentence( x ) for x in text1] )
    df['CleanedText']=text2
    print( df.head() )
    df.to_csv( f'clean_data/C_{file_name}.csv' )

clean_descriptions('Russianyt')