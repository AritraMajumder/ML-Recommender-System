import streamlit as st
import pickle 
import pandas as pd
import requests
import warnings
warnings.filterwarnings('ignore')

def poster(movie_id):
    response = requests.get("https://api.themoviedb.org/3/movie/{}?api_key=e11fecd2baad81b1ec908172ec33f192".format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w780"+data['poster_path']

sim = pickle.load(open('similarities.pkl','rb'))
def recommender(movie):
    ind = movies[movies['title']==movie].index[0]
    dist = list(enumerate(sim[ind]))
    rev = sorted(dist,reverse=True,key = lambda x:x[1])
    l = []
    k = []
    for i in rev[1:7]:
        movie_id = movies.iloc[i[0]].id
        l.append(movies.iloc[i[0]].title)
        k.append(poster(movie_id))
    return l,k


movies_dict = pickle.load(open('movies_dict.pkl','rb'))
movies = pd.DataFrame(movies_dict)
st.title('Movie Recommender System')

option = st.selectbox(
    'Enter a movie you liked',
    movies['title'].values)

if st.button('Recommend'):
    names,posters = recommender(option)
    import streamlit as st

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.text(names[0])
        st.image(posters[0])

    with col2:
        st.text(names[1])
        st.image(posters[1])

    with col3:
        st.text(names[2])
        st.image(posters[2])

    with col4:
        st.text(names[3])
        st.image(posters[3])

    with col5:
        st.text(names[4])
        st.image(posters[4])

    with col6:
        st.text(names[5])
        st.image(posters[5])