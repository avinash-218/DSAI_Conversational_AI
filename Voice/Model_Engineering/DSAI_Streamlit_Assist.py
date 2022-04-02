# Disclaimer:

#DeepSphere.AI developed these
#materials based on its team’s expertise
#and technical infrastructure, and we
#are sharing these materials strictly for
#learning and research. These learning
#resources may not work on other learning
#infrastructures and DeepSphere.AI
#advises the learners to use these materials
#at their own risk. As needed, we will be
#changing these materials without any
#notification and we have full ownership
#a1nd accountability to make any change
#to these materials.

#%%writefile app.py   #to run in notebook uncomment this

# Import Libraries
import streamlit as st
import random
import string
import nltk
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from gtts import gTTS
import os
import playsound
from PIL import Image
import speech_recognition as sr
import nltk
from nltk.stem import WordNetLemmatizer
warnings.filterwarnings('ignore')

# for downloading package files can be commented after First run
#nltk.download('popular', quiet=True)
#nltk.download('nps_chat',quiet=True)
#nltk.download('punkt') 
#nltk.download('wordnet')

class ChatBot:
    def __init__(self):
        #constructor
        self.vAR_posts = None #input corpus
        self.vAR_featuresets = None #extracted features from input corpus
        self.vAR_train_set = None
        self.vAR_test_set = None
        self.vAR_sent_tokens = None #sentence tokens
        self.vAR_word_tokens = None #word tokens
        self.vAR_remove_punct_dict = None
        self.vAR_lemmer = WordNetLemmatizer() #lemmatizer object

    def load_corpus(self):
        #load data from online chat services (from inbuilt package)
        self.vAR_posts = nltk.corpus.nps_chat.xml_posts()[:10000]

    def extract_featuresets(self):
        #function to extract features from input corpus
        self.vAR_featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in self.vAR_posts]

    def train_test_split(self):
        #split features into train and test data
        vAR_size = int(len(self.vAR_featuresets) * 0.1) #determine size for train-test split
        self.vAR_train_set, self.vAR_test_set = self.vAR_featuresets[vAR_size:], self.vAR_featuresets[:vAR_size] #split features into train and test data

    def train_classifier(self):
        #train classifier with train set
        self.vAR_classifier = nltk.NaiveBayesClassifier #NaiveBayes Classifier
        self.vAR_classifier = self.vAR_classifier.train(self.vAR_train_set) #train the classifier

    def tokenize(self, vAR_raw):
        # method to perform Tokenisation
        self.vAR_sent_tokens = nltk.sent_tokenize(vAR_raw) # converts to list of sentences 
        self.vAR_word_tokens = nltk.word_tokenize(vAR_raw) # converts to list of words

    def LemTokens(self, vAR_tokens):
        #function to lemmatize the input tokens
        #tokens - list of words to lemmatize
        return [self.vAR_lemmer.lemmatize(token) for token in vAR_tokens]

    def remove_punctuation(self):
        self.vAR_remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

    def LemNormalize(self, vAR_text):
        #function to remove punctuation and tokenize and then to lemmatize from input 
        # text - input corpus
        return self.LemTokens(nltk.word_tokenize(vAR_text.lower().translate(self.vAR_remove_punct_dict)))

    def response(self, vAR_user_response):
        # function to generate response and processing 
        # user_response - input response from user

        vAR_robo_response='' #chatbot response
        self.vAR_sent_tokens.append(vAR_user_response)
        vAR_TfidfVec = TfidfVectorizer(tokenizer=self.LemNormalize, stop_words='english') #object of TfidfVectorizer
        vAR_tfidf = vAR_TfidfVec.fit_transform(self.vAR_sent_tokens) #transform the english text to vector
        vAR_vals = cosine_similarity(vAR_tfidf[-1], vAR_tfidf) #find similarity by cosine angle
        vAR_idx = vAR_vals.argsort()[0][-2]
        vAR_flat = vAR_vals.flatten()
        vAR_flat.sort()
        vAR_req_tfidf = vAR_flat[-2]
        if(vAR_req_tfidf == 0): #if input can't be recognized
            vAR_robo_response = vAR_robo_response+"I am sorry! I don't understand you"
            return vAR_robo_response
        else:
            vAR_robo_response = vAR_robo_response+self.vAR_sent_tokens[vAR_idx]
            return vAR_robo_response

    def chat(self):
        # method to chat with robot
        vAR_r1 = random.randint(1,10000000)
        vAR_r2 = random.randint(1,10000000)
        vAR_file = str(vAR_r2)+"randomtext"+str(vAR_r1) +".mp3" #generate random file name

        #Recording voice input using microphone 
        vAR_flag = True
        fst="Hello, i am your personal chatbot. I will answer your queries. If you want to exit, say Bye"
        vAR_tts = gTTS(fst,lang="en",tld="com") #google text to speech API to convert the message to audio
        vAR_tts.save(vAR_file) #save the audio with the random filename generated

        vAR_r = sr.Recognizer() #recognize user input
        st.write(f'<p style="font-family: sans-serif;font-size: 15px;text-transform: capitalize;background-color: #d9d8d8;padding: 18px;border-radius: 15px">{fst}</p>', unsafe_allow_html=True)
        playsound.playsound(vAR_file,True) #play the bot reply
        os.remove(vAR_file) #remove the file generated

        # Taking voice input and processing
        while(vAR_flag==True):
            with sr.Microphone(device_index=1) as source: #microphone as input device
                st.write("Listening...")
                vAR_audio= vAR_r.adjust_for_ambient_noise(source)
                vAR_audio = vAR_r.listen(source)
            try:
                vAR_user_response = format(vAR_r.recognize_google(vAR_audio))
                st.write(f'<p style="font-family: sans-serif;color: white;font-size: 15px;text-align:right;text-transform: capitalize;background-color: #190380;padding: 18px;border-radius: 15px">{vAR_user_response}</p>', unsafe_allow_html=True)

                vAR_clas = self.vAR_classifier.classify(dialogue_act_features(vAR_user_response)) #classify recognized input
                if(vAR_clas!='Bye'):
                    if(vAR_clas=='Emotion'):
                        vAR_flag=False
                        st.write("Bot: You are welcome..")
                    else:
                        st.write("Bot:")
                        vAR_res=(self.response(vAR_user_response))
                        st.write(f'<p style="font-family: sans-serif;font-size: 15px;text-transform: capitalize;background-color: #d9d8d8;padding: 18px;border-radius: 15px">{vAR_res}</p>', unsafe_allow_html=True)
                        self.vAR_sent_tokens.remove(vAR_user_response)
                        vAR_tts = gTTS(vAR_res,tld="com")
                        vAR_tts.save(vAR_file)
                        playsound.playsound(vAR_file,True)
                        os.remove(vAR_file)
                else:
                    vAR_flag=False
                    st.write("Bot: Bye! take care..")
            except sr.UnknownValueError:
                st.write("Oops! Didn't catch that")
                pass

    def RUN_BOT(self, vAR_raw):
        #method to run all methods
        self.load_corpus()
        self.extract_featuresets()
        self.train_test_split()
        self.train_classifier()
        self.tokenize(vAR_raw)
        self.remove_punctuation()

def local_css(vAR_file_name):
    #function to apply style formatting from styles.css file in streamlit
    #filename - css file contains webpage formatting options
        with open(vAR_file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("../User_Interface/style.css") #function call to load .css file and apply in streamlit webpage

vAR_col1, vAR_col2, vAR_col3 = st.columns([1,1,1])
vAR_col2.image('../User_Interface/DSAI_Logo.jpg', use_column_width=True)

st.title("Voice Based Chatbot") #set title for webpage

#Reading in the input_corpus
with open('../Utility/DSAI_Deepsphere_Bot_Reply','r', encoding='utf8', errors ='ignore') as fin:
    vAR_raw = fin.read().lower()

def dialogue_act_features(post):
    # To Recognise input type as QUES. 
        vAR_features = {}
        for word in nltk.word_tokenize(post):
            vAR_features['contains({})'.format(word.lower())] = True
        return vAR_features

vAR_chatbot_object = ChatBot()
vAR_chatbot_object.RUN_BOT(vAR_raw)

if st.button("Get your Assistant"): #event listener for 'Get your Assistant' button
    vAR_chatbot_object.chat()

image2 = Image.open('../User_Interface/DSAI_Robot.png')
st.image(image2)


#Copyright Notice:

#Local and international copyright laws protect
#this material. Repurposing or reproducing
#this material without written approval from
#DeepSphere.AI violates the law.

#(c) DeepSphere.AI