#extract tags
import ast
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def tagger(a):
    l = []
    for i in ast.literal_eval(a):
        l.append(i['name'])
    return l

def tagger3(a):
    l = []
    count = 0
    for i in ast.literal_eval(a):
        if count!=3:
            l.append(i['name'])
            count+=1
        else:
            break
    return l
    
def tagger_direct(a):
    l = []
    for i in ast.literal_eval(a):
        if i['job']=='Director':
            l.append(i['name'])
    return l

def stemmer(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

