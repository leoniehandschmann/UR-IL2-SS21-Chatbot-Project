import random
import json
import pickle
from tkinter import messagebox

import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

from tkinter import *

from PIL import ImageTk, Image

lemmatizer = WordNetLemmatizer()

# laden der gesamten Daten
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# Funktion um den Satz aufzubereiten(tokenizing,lemmatize)
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Funktion gibt mit einem bag of word an ob das Wort enthalten ist (1) oder nicht (0)
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Funktion gibt die wahrscheinlichste Kategorie passend zum Wort zurück
# die Kategorie mit der höchsten Wahrscheinlichkeit steht an erster Stelle der return_list
def predict_class(sentence):
    bag = bag_of_words(sentence)
    res = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Funktion lädt die Antworten des Chatbots
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            if i['tag'] == "images_memes":
                image_response(random.choice(i['responses']))
            if i['tag'] == "images_animals":
                image_response(random.choice(i['responses']))
            if i['tag'] == "images_random":
                image_response(random.choice(i['responses']))
            break
    return result

# Funktion gibt einen neuen Canvas mit einem Bild zurück
def image_response(file_path):
    root = Toplevel()

    img = ImageTk.PhotoImage(Image.open(file_path))
    panel = Label(root, image=img)
    panel.pack(side="bottom", fill="both", expand="yes")

    root.protocol("WM_DELETE_WINDOW", lambda arg = root: new_image(arg))
    root.geometry("500x500")
    root.mainloop()

#Funktion initialisiert eine Messagebox --> Abfrage ob ein neues Bild gezeigt werden soll --> Canvas erstellen
def new_image(root):
    root.destroy()
    if messagebox.askyesno("Image ask","Do you want to see another picture?"):

        for i in intents["intents"]:
            for pattern in i["patterns"]:
                if "random" == pattern:
                    w = i["responses"]

        baseRoot = Toplevel()
        img = ImageTk.PhotoImage(Image.open(random.choice(w)))
        panel = Label(baseRoot, image=img)
        panel.pack(side="bottom", fill="both", expand="yes")
        baseRoot.geometry("500x500")
        baseRoot.protocol("WM_DELETE_WINDOW", lambda arg=baseRoot: new_image(arg))
        baseRoot.mainloop()

    ChatLog.insert(END, "Bot: " + "How do you feel now?" + '\n\n')



# Funktion ist verantwortlich für die Antworten des Chatbots
def chatbot_response(text):
    ints = predict_class(text)
    res = get_response(ints, intents)
    return res

#Goodbye Nachricht des Chatbots mit Hilfe einer MessageBox
def on_close():
    if messagebox.askquestion("Goodbye", "Do you really want to go? Ok goodbye,hope to see you soon!"):
        base.destroy()

# Dialogfenster des Chatbots wird erstellt mit Tkinter
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

#Initialisierung des Chat Fensters
base = Tk()
base.title("HappyBot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)

#Scrollbar
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
#Send Button
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )
#Eingabebox
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#Platzierung der Elemente im Fenster
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.protocol("WM_DELETE_WINDOW", on_close)
base.mainloop()



#print("GO,Bot is running!")