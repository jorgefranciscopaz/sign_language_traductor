import pyttsx3

voz = pyttsx3.init()
for v in voz.getProperty('voices'):
    print(f"ID: {v.id}\nNombre: {v.name}\nIdioma: {v.languages}\n")
