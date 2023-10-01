import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


ps = PorterStemmer()
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    L=[]
    for i in text:
        if(i.isalnum()):
            L.append(i)
    text=L[:]
    L.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            L.append(i)
    text=L[:]
    L.clear()
    for i in text:
        L.append(ps.stem(i))
    return " ".join(L)
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("SMS spam classifier")

input_text = st.text_area(
        "Enter the message"
    )
if st.button('Predict'):

    #1. preprocess
    transformed_text = transform_text(input_text)
    #2. vectorize
    vectorized_text = tfidf.transform([transformed_text])
    #3. predict
    result = model.predict(vectorized_text)[0]
    #4. display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
