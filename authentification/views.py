from django.shortcuts import render, redirect, HttpResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth import get_user_model
from authentification.models import Utilisateur
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


# Mes graphiques
fig = go.Figure()
scatter = go.Scatter(x=[0,1,2,3], y=[0,1,2,3],
                     mode='lines', name='test',
                     opacity=0.8, marker_color='green')
fig.add_trace(scatter)
plt_div = plot(fig, output_type='div')

df2 = px.data.iris() # iris is a pandas DataFrame
fig2 = px.scatter(df2, x="sepal_width", y="sepal_length", title="Scatter plot")
graph2 = plot(fig2, output_type='div')


df3 = px.data.tips()
fig3 = px.box(df3, x="time", y="total_bill", title="Boîte à moustache")
graph3 = plot(fig3, output_type='div')

z = [[.1, .3, .5, .7, .9],
     [1, .8, .6, .4, .2],
     [.2, 0, .5, .7, .9],
     [.9, .8, .4, .2, 0],
     [.3, .4, .5, .7, 1]]

fig4 = px.imshow(z, text_auto=True)
graph4 = plot(fig4, output_type='div')


import plotly.express as px

df = px.data.tips()
fig = px.scatter(
    df, x='total_bill', y='tip', opacity=0.65,
    trendline='ols', trendline_color_override='darkblue'
)
#fig.show()


# Clustering
# DBScan

dfPenguins = pd.read_csv("media/penguins.csv")
X = dfPenguins[dfPenguins.describe().columns].dropna().values

X = StandardScaler().fit_transform(X)

db = DBSCAN().fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
#Homogeneity = metrics.homogeneity_score(labels_true, labels)



#Clustering
#PREPRO
data = pd.read_csv("media/penguins.csv")
data[data['sex']=='.']
data.loc[336,'sex'] = 'FEMALE'

data['species']=data['species'].map({'Adelie':0,'Gentoo':1,'Chinstrap':2})

dummies = pd.get_dummies(data[['island','sex']],drop_first=True)
# we do not standardize dummy variables 
df_to_be_scaled = data.drop(['island','sex'],axis=1)
target = df_to_be_scaled.species
df_feat= df_to_be_scaled.drop('species',axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_feat)
df_scaled = scaler.transform(df_feat)
df_scaled = pd.DataFrame(df_scaled,columns=df_feat.columns[:4])
df_preprocessed = pd.concat([df_scaled,dummies,target],axis=1)
df_preprocessed.head()

df_preprocessed = df_preprocessed.dropna()

from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

kmeans = KMeans(3,init='k-means++')
kmeans.fit(df_preprocessed.drop('species',axis=1))


accu = {np.round(100*accuracy_score(df_preprocessed.species,kmeans.labels_),2)}








from .models import FilesUpload

def home(request):
    if request.method == "POST":
        file2 = request.FILES["file"]
        document = FilesUpload.objects.create(file=file2)
        document.save()
        return HttpResponse("Your file uploaded")
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
    return render(request, "inscription.html", {"message" : message})

def connexion(request):
    # La méthode POSt est utilisé quand des infos
    # sont envoyées au back-end
    # Autrement dit, on a appuyé sur le bouton
    # submit
    message = ""
    if request.method == "POST":
        identifiant = request.POST["identifiant"]
        motdepasse = request.POST["motdepasse"]
        utilisateur = authenticate(username = identifiant,
                                   password = motdepasse)
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
    context = {"n_clusters_" : n_clusters_,
               "n_noise_" : n_noise_,
               "graphique": plt_div,
               "graph2": graph2,
               "graph3": graph3,
               "graph4": graph4
               }
    return render(request, "index.html", context)

def regression(request):
    return render(request, "regression.html")


def clustering(request):
    context = {"n_clusters_" : n_clusters_,
               "accu": accu }
    return render(request, "clustering.html", context)

def classification(request):
    return render(request, "classification.html")