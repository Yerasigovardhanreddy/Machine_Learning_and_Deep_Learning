import streamlit as st
from pickle import load
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#################################################################


def clean(doc):
    
    # Removing Special characters
    regex = r'[^a-zA-Z\s]'
    doc = re.sub(regex, '', doc)

    # Lowercase text
    text = doc.lower()

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Stop word removal
    stop_words = set(stopwords.words('english'))
    custom_stop_words = {'br','one','product','will','flavor','love','taste'}
    stop_words = stop_words.union(custom_stop_words)
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    # Join and return
    return " ".join(lemmatized_tokens)


def tokenizer(doc):
    # Tokenization
    return nltk.word_tokenize(doc) 


# Loading pretrained CountVectorizer from pickle file
vectorizer = load(open('models/countvectorizer.pkl', 'rb'))
    
# Loading pretrained logistic classifier from pickle file
classifier = load(open('models/logit_model.pkl', 'rb'))


def predict(tweet):
    
    # Converting text to numerical vector
    clean_tweet_encoded = vectorizer.transform([tweet])
    
    # Prediction
    prediction = classifier.predict(clean_tweet_encoded)
    
    return prediction

#################################################################

st.title('Amazon Fine Food Review')

st.image(r"img\amazon_dog_food.jpeg",width = 300)

review = st.text_area('Enter your Review')

sbmt_button = st.button('Submit')

if sbmt_button == True and review:
    sentiment = predict(review)
    if (sentiment  <= 3):
        st.snow()
        st.write("The review provided by the user is Negative")
    else:
        st.balloons()
        st.write("The review provided by the user is Positive")

