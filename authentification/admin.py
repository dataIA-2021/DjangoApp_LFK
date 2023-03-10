from django.contrib import admin
from authentification.models import Utilisateur

# Afficher toutes les colonnes
class colonnesTableUtilisateur(admin.ModelAdmin):
    list_display = [field.name for field in Utilisateur._meta.fields]

# J'attache mon model Utilisateur à la page d'administration
# pour pouvoir Create-Read-Update-Delete dessus
admin.site.register(Utilisateur, colonnesTableUtilisateur)


from .models import FilesUpload
admin.site.register(FilesUpload)