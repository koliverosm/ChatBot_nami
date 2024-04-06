import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import pickle as abrir

# Configuración inicial
lemmatizer = WordNetLemmatizer()
categorias = json.loads(open('../Nami/patrones/categorias.json').read())
symbol_ignore = ['?', '!', '¿', '.', ',']

# Preprocesamiento de datos
palabras = []
classes = []
doc_category = []
for categoria in categorias['categorias']:
    for pattern in categoria['patterns']:
        word_list = nltk.word_tokenize(pattern)
        palabras.extend([lemmatizer.lemmatize(word.lower())
                        for word in word_list if word not in symbol_ignore])
        doc_category.append((word_list, categoria["tag"]))
        if categoria["tag"] not in classes:
            classes.append(categoria["tag"])
palabras = sorted(set(palabras))

# Guardar palabras y clases
with open('../Nami/palabras/palabras.pkl', 'wb') as archivo:
    abrir.dump(palabras, archivo)
with open('../Nami/clases/classes.pkl', 'wb') as archivo:
    abrir.dump(classes, archivo)

entrenamiento_x = []
entrenamiento_y = []
for category in doc_category:
    bag = np.array([1 if lemmatizer.lemmatize(word.lower())
                   in category[0] else 0 for word in palabras])
    salida_fila = np.zeros(len(classes))
    salida_fila[classes.index(category[1])] = 1
    entrenamiento_x.append(bag)
    entrenamiento_y.append(salida_fila)

entrenamiento_x = np.array(entrenamiento_x)
entrenamiento_y = np.array(entrenamiento_y)

# Crear y compilar el modelo
model = Sequential([
    Dense(128, input_shape=(len(palabras),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer=SGD(
    learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])

# Entrenar y guardar el modelo
model.fit(entrenamiento_x, entrenamiento_y, epochs=1000, batch_size=5, verbose=1 )
model.save("../Nami/modelo/initial_model.h5")
