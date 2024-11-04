import pandas as pd
import numpy as np
import json
from util import *

import warnings
warnings.filterwarnings('ignore')

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

#both dfs have title column, concat movie details and cast
#details of the same title

movies = movies.merge(credits,on='title')
movies = movies[['genres','id','keywords','title','overview','cast','crew']]

movies.dropna(inplace=True)

movies['genres'] = movies['genres'].apply(tagger)
movies['keywords'] = movies['keywords'].apply(tagger)
movies['cast']=movies['cast'].apply(tagger3)
movies['crew'] = movies['crew'].apply(tagger_direct)

movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tag'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

m2 = movies[['id','title','tag']]
m2['tag'] = m2['tag'].apply(lambda x:" ".join(x))
m2['tag'] = m2['tag'].apply(lambda x:x.lower())

#print(m2['tag'][0]) #check progress

#stemming
m2['tag'] = m2['tag'].apply(stemmer)

#count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000,stop_words = 'english')

vect = cv.fit_transform(m2['tag']).toarray()

#cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(vect)


def recommender(movie):
    ind = m2[m2['title']==movie].index[0]
    dist = list(enumerate(sim[ind]))
    rev = sorted(dist,reverse=True,key = lambda x:x[1])
    for i in rev[1:6]:  #tweak upper limit for no of recommendations
        print(i,"\t",m2.iloc[i[0]].title)


recommender('Spectre')

import pickle
#main files
pickle.dump(m2.to_dict(),open('movies_dict.pkl','wb'))
pickle.dump(sim,open('similarities.pkl','wb'))