from django.db import models
from django.contrib.auth.models import AbstractUser
from django import forms

class Utilisateur(AbstractUser):
    photo = models.ImageField()

class FilesUpload(models.Model):
    file = models.FileField()

