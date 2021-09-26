
import random # picking random response from intents
import json # to read intents.json
import pickle # for serialization
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


lemmatizer = WordNetLemmatizer

# einlesen der intents file
intents = json.loads(open('intents.json').read())

# einzelne Wörter
words = []
# Kategorien wie z.B. greetings
classes = []
# Zusammenpassende Wörter und Kategorien
documents = []
ignore_letters = ['?', '!', '.', ',']

# alle patterns in der Intents Datei werden tokenized,
# diese einzelnen tokenized Wörter werden der words Liste hinzugefügt
# die Kategorien werden den classes hinzugefügt, documents wird hier ebensfalls befüllt
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatizing der words list
words = [lemmatizer.lemmatize(lemmatizer, word) for word in words if word not in ignore_letters]
# sortiert wird nach dem Alphabet; Duplikate werden entfernt
words = sorted(set(words))

classes = sorted(set(classes))

# die Listen werden hier als pickle file gespeichert
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
# template von Nullen
output_empty = [0] * len(classes)

# für jedes Element in documents wird ein bag of words erstellt
# ist das Wort in documents enthalten wird bag eine 1 hinzugefügt, wenn nicht dann 0
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(lemmatizer,word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    #training list wird gefüllt mit den Daten aus der documents Liste
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# randomisierte Data wird als Numpy Array gespeichert
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Neural Network Model wird erstellt
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# as many neurals as classes
model.add(Dense(len(train_y[0]), activation='softmax'))

# optimizer erstellen
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# Modell kompilieren
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Modell speichern
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)

print("Done loading training data!")
