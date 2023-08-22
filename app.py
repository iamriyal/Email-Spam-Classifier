import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('SMS Spam Predicter')
input_sms = st.text_area('Enter Your Message')
if st.button('Predict'):
# preprocessing
    transformed_sms=transform_text(input_sms)
# vectorize
    vector_input=tfidf.transform([transformed_sms])
# predict
    result=model.predict(vector_input)[0]
# Display
    if result==1:
        st.header('spam')
    else:
        st.header('Not Spam')







