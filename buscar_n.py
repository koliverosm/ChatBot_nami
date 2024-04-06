import spacy as sp
import es_core_news_sm
nlp = sp.load("es_core_news_sm")   
def capturar_nombre(frase):
    doc = nlp(frase)
    for entidad in doc.ents:
        if entidad.label_ == "PER":
            return entidad.text.lower()  # Devuelve el nombre en minúsculas si encuentra una entidad de tipo "PER"

    return None  # Devuelve None si no se encontró un nombre

# Ejemplo de uso
frase = "Me llamo Kevin Kevin"
nombre = capturar_nombre(frase)
print("Nombre capturado:", nombre)