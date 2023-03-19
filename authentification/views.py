from .models import FilesUpload
from django.shortcuts import render, redirect, HttpResponse
from django.http import HttpResponseRedirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from authentification.models import Utilisateur
from .forms import LabelForm
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
from sklearn.metrics import confusion_matrix, f1_score

import plotly.express as px


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
    f1_scores = []
    
    X = preprocessing(data.loc[:, data.columns != label])
    y = data[label]
    labels = pd.unique(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify = y)
    
    for model in models:
        model.fit(X_train, y_train)
        trained_models.append(model)
            
        y_pred = model.predict(X_test)
        f1_scores.append(f1_score(y_test, y_pred, labels = labels, average = 'weighted'))
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrixs.append(conf_matrix)

    return(trained_models, conf_matrixs, labels, f1_scores)

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
    # data = fichier()
    # context={"features": data.columns.to_list() }
    if request.method == "POST":
        form = LabelForm(request.POST)
        if form.is_valid():
            # print("ici")
            # print(form.cleaned_data['label'])
            value = form.cleaned_data['label']
            global val
            def val():
                return value
            
            # if form.is_valid():
            #     form.save()
            # return render(request, 'classification_results.html', {'form': form})
            return redirect("classification_results")
    else:
        form = LabelForm()
    return render(request, 'classification.html', {'form': form})



def classification_results(request):
    
    data = fichier()

    #Get Label
    label = val()
    #Train Models
    models = classification_training(data, label)

    values = models[2]

    fig1 = px.imshow(models[1][0], labels = dict(x="DecisionTreeClassifier"), x = values, y = values, text_auto=True)
    graphique = plot(fig1, output_type='div')

    fig2 = px.imshow(models[1][1], labels = dict(x="RandomForestClassifier"), x = values, y = values, text_auto=True)
    graph2 = plot(fig2, output_type='div')

    fig3 = px.imshow(models[1][2], labels = dict(x="SVC"), x = values, y = values, text_auto=True)
    graph3 = plot(fig3, output_type='div')

    fig4 = px.bar(models[3])
    graph4 = plot(fig4, output_type='div')

    context = {
        "graphique": graphique,
        "graph2": graph2,
        "graph3": graph3,
        "graph4": graph4
        }

    return render(request, "classification_results.html", context)

