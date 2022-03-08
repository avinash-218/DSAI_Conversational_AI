import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from tensorflow import keras
#from keras.models import load_model
import json
import random
#Creating GUI with tkinter
import tkinter
from tkinter import *

model = keras.models.load_model('../Utility/DSAI_Chatbot_Model.h5') #load model
intents = json.loads(open('../Utility/DSAI_Intents.json').read()) #load json file
words = pickle.load(open('../Utility/DSAI_Words.pkl','rb')) #load saved words object (pickle)
classes = pickle.load(open('../Utility/DSAI_Classes.pkl','rb')) #load saved classes object (pickle)

lemmatizer = WordNetLemmatizer() #lemmatizer

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
    #sentence - input message from EntryBox
    #model - saved model by which class is to be predicted
    p = bow(sentence, words,show_details=False) #extract bag of words
    res = model.predict(np.array([p]))[0] #predict class
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

def chatbot_response(msg):
    # predicts class and getResponse for the input message
    # msg - input message from EntryBox
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

def send():
    #event listener for send button
    msg = EntryBox.get("1.0",'end-1c').strip() #get message from entry box
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n') #display input message in the chatlog
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg) #get response for the input message
        ChatLog.insert(END, "Bot: " + res + '\n\n') #display response message in the chatlog

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


base = Tk() #creates tinker object
base.title("Hello") #window title
base.geometry("400x500") #window dimension
base.resizable(width=FALSE, height=FALSE) #window is not resizable

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send ) #send function is event loop

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)

#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()