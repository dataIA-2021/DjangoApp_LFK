from .models import FilesUpload
from django.shortcuts import render, redirect, HttpResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from authentification.models import Utilisateur, Label
from plotly.offline import plot
import plotly.graph_objs as go
from sklearn.datasets import make_moons
import plotly.express as px
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


def nettoyage(type=['csv', 'h5']):
    if type == 'csv':
        clean = []
        for (directory, underdir, files) in os.walk(r'.\media'):
            clean.extend(files)
        for l in clean:
            if l.endswith(('.csv', '.xls')):
                os.remove('.\media\\'+l)
    else:
        clean = []
        for (directory, underdir, files) in os.walk(r'.\media'):
            clean.extend(files)
        for l in clean:
            if l.endswith('.h5'):
                os.remove('.\media\\'+l)

def fichier():
    file = list()
    for (directory, underdir, files) in os.walk(r'.\media'):
        file.extend(files)

    data = None
    for i in file:
        if i.endswith((".csv", ".xls")):
            data = pd.read_csv('.\media\\'+i)
    return(data)

def preprocessing(data_to_preprocess):
    
    numerical_datas = data_to_preprocess[data_to_preprocess.select_dtypes(
        include=['int', 'float']).columns].fillna(0).values
    return (numerical_datas)

def classification_training(data, label):
    
    models  = [DecisionTreeClassifier(), RandomForestClassifier(), SVC()]
    trained_models = []
    conf_matrixs = []
    
    X = preprocessing(data.loc[:, data.columns != label])
    y = data[label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y)
    
    for model in models:
        model.fit(X_train, y_train)
        trained_models.append(model)
            
        y_pred = model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrixs.append(conf_matrix)

    return(trained_models, conf_matrixs)

def home(request):
    return render(request, "index.html")

def inscription(request):
    message = ""
    if request.method == "POST":
        if request.POST["motdepasse1"] == request.POST["motdepasse2"]:
            modelUtilisaleur = get_user_model()
            identifiant = request.POST["identifiant"]
            motdepasse = request.POST["motdepasse1"]
            utilisateur = modelUtilisaleur.objects.create_user(username=identifiant,
                                                               password=motdepasse)
            return redirect("connexion")
        else:
            message = "⚠️ Les deux mots de passe ne concordent pas ⚠️"
    return render(request, "inscription.html", {"message": message})


def connexion(request):
    # La méthode POSt est utilisé quand des infos
    # sont envoyées au back-end
    # Autrement dit, on a appuyé sur le bouton
    # submit
    message = ""
    if request.method == "POST":
        identifiant = request.POST["identifiant"]
        motdepasse = request.POST["motdepasse"]
        utilisateur = authenticate(username=identifiant,
                                   password=motdepasse)
        if utilisateur is not None:
            login(request, utilisateur)
            return redirect("index")
        else:
            message = "Identifiant ou mot de passe incorrect"
            return render(request, "connexion.html", {"message": message})
    # Notre else signifie qu'on vient d'arriver
    # sur la page, on a pas encore appuyé sur le
    # bouton submit
    else:
        return render(request, "connexion.html")


def deconnexion(request):
    logout(request)
    return redirect("connexion")


def suppression(request, id):
    utilisateur = Utilisateur.objects.get(id=id)
    logout(request)
    utilisateur.delete()
    return redirect("connexion")


@login_required
def index(request):
    if request.method == "POST":
        nettoyage("csv")
        file2 = request.FILES["file"]
        document = FilesUpload.objects.create(file=file2)
        document.save()
        return redirect('classification')
    return render(request, "index.html")


def regression(request):
    return render(request, "regression.html")

def clustering(request):
    return render(request, "clustering.html")


def classification(request):
    #File Upload
    form = Label()
    
    if request.method == "POST":
        return render(request, 'classification_results.html', {'form': form})
        # return redirect('classification_results')
    
    return render(request, 'classification.html', {'form': form})


def classification_results(request):
    
    data = fichier()

    #Get Label
    label = form
    #Train Models
    models = classification_training(data, label)
        
    return render(request, "classification_results.html")

