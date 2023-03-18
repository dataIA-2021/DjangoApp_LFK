from django.db import models
from django.contrib.auth.models import AbstractUser
from django import forms

class Utilisateur(AbstractUser):
    photo = models.ImageField()

class FilesUpload(models.Model):
    file = models.FileField()

class Label(forms.Form):
   label = forms.CharField(max_length=1000)
