from django.db import models
from django.contrib.auth.models import AbstractUser

class Utilisateur(AbstractUser):
    photo = models.ImageField()

class FilesUpload(models.Model):
    file = models.FileField()
