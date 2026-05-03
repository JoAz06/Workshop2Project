from django import forms

class MedicalForm(forms.Form):
    total_claims_paid = forms.FloatField()
    avg_claim_amount = forms.FloatField()
    risk_score = forms.FloatField()
    ldl = forms.FloatField()
    income = forms.FloatField()
    provider_quality = forms.FloatField()
    bmi = forms.FloatField()
    hba1c = forms.FloatField()
    systolic_bp = forms.FloatField()
    diastolic_bp = forms.FloatField()
    hospitalizations_last_3yrs = forms.IntegerField()
    age = forms.IntegerField()
    sex = forms.ChoiceField(choices=[("Male","Male"),("Female","Female")])