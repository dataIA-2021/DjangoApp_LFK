from django import forms

class LabelForm(forms.Form):
   label = forms.CharField(max_length = 40, required = True)