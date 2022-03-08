import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from tensorflow import keras
#from keras.models import load_model
from flask import Flask, render_template, request
import json
import random

app = Flask(__name__)

lemmatizer = WordNetLemmatizer() #lemmatizer

model = keras.models.load_model('../Utility/DSAI_Chatbot_Model.h5') #load model
intents = json.loads(open('../Utility/DSAI_Intents.json').read())#load json file
words = pickle.load(open('../Utility/DSAI_Words.pkl','rb'))#load saved words object (pickle)
classes = pickle.load(open('../Utility/DSAI_Classes.pkl','rb'))#load saved classes object (pickle)

@app.route("/")
def index():
    return render_template("DSAI_Index.html")

def clean_up_sentence(sentence):
    # tokenize and lemmatize input sentence
    #sentence - input message from entry box
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    # get response for the input message after predicting class
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

@app.route('/get')
def chatbot_response():
    message = request.args.get('msg')
    userText = predict_class(message, model)
    res = getResponse(userText, intents)
    return res

if __name__ == "__main__":
    app.run(debug=True)