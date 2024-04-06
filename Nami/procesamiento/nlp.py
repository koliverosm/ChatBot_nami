import es_core_news_sm
import numpy as np
import random
import nltk
from ..models.resourse_name import objeto_usuario
import spacy
import re
user = objeto_usuario()


class Preprocessing:
    def __init__(self):
        self.lemmatizer = nltk.WordNetLemmatizer()

    def capitalizar_primeras_letras(self, frase):
        palabras = frase.split()
        frase_capitalizada = ' '.join(
            [palabra.capitalize() for palabra in palabras])
        pro = CapturarNombre()

        return pro.process_message(frase_capitalizada)

    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(
            word.lower()) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence, words):
        sentence_words = self.clean_up_sentence(sentence)
        bag = np.array([1 if word in sentence_words else 0 for word in words])
        return bag

    @staticmethod
    def predict_class(sentence, model, words, classes):
        preprocessing = Preprocessing()
        bow = preprocessing.bag_of_words(sentence, words)
        res = model.predict(np.array([bow]))[0]
        max_index = np.argmax(res)
        category = classes[max_index]
        return category

    @staticmethod
    def get_response(ints, intents_json):
        list_of_intents = intents_json['categorias']
        for intent in list_of_intents:
            if intent['tag'] == ints:
                result = random.choice(intent['responses'])

        return result


nlp = es_core_news_sm.load()

respuestas_con_nombre = [
    "Hola, {nombre}! ¿En qué puedo ayudarte hoy?",
    "¡Hola, {nombre}! ¿Cómo puedo ayudarte?",
    "¿Qué tal, {nombre}? ¿En qué puedo ayudarte?",
    "¡Hola, {nombre}! ¿Cómo estás?",
    "¡Hola! ¿En qué puedo ayudarte hoy, {nombre}?"
]


class CapturarNombre:

    def process_message(self, oracion):
        doc = nlp(oracion.lower())  # Normaliza la oración a minúsculas
        nombre = None

        # Expresión regular para buscar nombres en la forma "Me llamo [nombre]"
        pattern = r"(me llamo|mi nombre es|yo soy|soy) (\w+)"
        match = re.search(pattern, oracion.lower())
        if match:
            nombre = match.group(2)  # Captura el nombre

        # Buscar nombres completos (nombre y apellido)
        for ent in doc.ents:
            if ent.label_ == "PER":
                nombre_completo = ent.text.split()
                if len(nombre_completo) > 1:
                    # Captura el nombre completo
                    nombre = " ".join(nombre_completo)
                    break

        if nombre:
            nombre_c = nombre.capitalize()
            return  CapturarNombre.obtener_respuesta_con_nombre(nombre_c) # Devuelve el nombre con la primera letra en mayúscula
        else:
            return None  # Devuelve None si no se encontró un nombre

    @staticmethod
    def obtener_respuesta_con_nombre(nombre:str):
        # Selecciona una respuesta aleatoria y reemplaza el marcador de posición con el nombre
        respuesta = random.choice(respuestas_con_nombre).format(nombre=nombre)
        return respuesta
