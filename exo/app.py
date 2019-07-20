from flask import Flask, request
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump, load
import json
from utils import parse_box_office  #Import de la fonction stockée dans le folder utils.


app = Flask(__name__)

'''Endpoint permettant de récupérer les metadata d'une liste de films'''
@app.route('/get_movie_metadata', methods=['POST'])  #Déclaration du endpoint de l'API
def get_movie_metadata():  #Déclaration de la fonction remplis par l'API.
    movies = json.loads(request.form['movies'])  #Récupération des arguments passés dans l'URL, puis conversion en JSON.

    '''Passage de la liste de films (récupérée dans l'URL) dnas l'API OMDB pour récupérer les metadonnées'''
    return json.dumps([   #JSON.dumps permet de remettre au format liste un fichier JSON.
        requests.get('http://www.omdbapi.com/?t={}&apikey=f6adbb42'.format(movie)).json()
        for movie in movies
    ])


'''Endpoint permettant d'entrainet un modèle de machine learning sur une liste de films avec leurs metadata'''
@app.route('/train_model', methods=['POST'])
def train_model():
    model_name = request.form['model_name']  #Récupération du choix du model dans l'URL de l'API.
    data = json.loads(request.form['data'])  #Récupération du dataset (au format liste) dans l'URL d'appel
    df = pd.DataFrame(data)  #Conversion des data en DataFrame

    '''Data preparation pour optimiser le modèle de learning'''
    df['BoxOffice'] = df['BoxOffice'].apply(parse_box_office).dropna()  #Conversion des montants $ en Float à l'aide de la fonction importée
    df['USA?'] = df['Country'].apply(lambda country: 1 if (country == 'USA') else 0)  #Remplacement des Country en 1 si USA, O si autre
    df['Genre'] = df['Genre'].apply(lambda genre: genre.split(',')[0])  #Remplacement de la liste des genres par le premier genre de la liste
    df = pd.get_dummies(df, columns=['Genre'], drop_first=True)  #Get.dummies sur les genres
    df = df.dropna()  #Suppression des NA dans le Dataset

    X = df[['BoxOffice', 'USA?'] + [col for col in df.columns if 'Genre_' in col]]  #Construction de la variable X en ne gardant que les colonnes intéressantes
    y = df['imdbRating'].astype(float)  #Création de la variable Y au format float

    reg = LinearRegression()  #Instanciation de la RegLin

    reg.fit(X, y)  #Entrainement du modèle

    dump(reg, model_name)  #Enregistrement du modèle dans un fichier sur le serveur (l'ordinateur en l'occurence).

    return 'SUCCESS'  #Renvoyer le message de succès si le modèle est bien entrainé, et qu'il est bien enregistré


'''Endpoint permettant de prédire le score IMDB d'un film (ou liste de films) grace au modèle entrainé au dessus'''
@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model_name']
    data = json.loads(request.form['data'])
    df = pd.DataFrame(data)

    return json.dumps(load(model_name).predict(df).tolist())



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

