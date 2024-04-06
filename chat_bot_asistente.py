import json
from keras.models import load_model
import pickle
from Nami.procesamiento.nlp import Preprocessing
from Nami.procesamiento import nlp

search_name = nlp.CapturarNombre()

# Cargar Los Modulos PreProsesados
intents = json.loads(open('./Nami/patrones/categorias.json').read())
words = pickle.load(open('./Nami/palabras/palabras.pkl', 'rb'))
classes = pickle.load(open('./Nami/clases/classes.pkl', 'rb'))
model = load_model('./Nami/modelo/initial_model.h5')

#Funcion Que generar El PDF
def view_pdf(response: str):

    return "view_pdf"

#While Que mantiene el Chat En Linea
while True:
    message = input("Tu: ")
    intent = Preprocessing.predict_class(message, model, words, classes)
    response = Preprocessing.get_response(intent, intents)


    if response == "crear volante pdf":
        response_ = view_pdf(response)
        print("Respuesta: ", response_)
    elif response == "nombre usuario":
       
        print( search_name.process_message(message))
    else:
        print("Respuesta General: ", response)
