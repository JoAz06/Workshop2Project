from django.shortcuts import render
from  .forms import MedicalForm, RiskPredictionForm
from .ml_model import predict_if_high_risk, predict_medical_cost

def home(request):
    return render(request, "core/index.html")

def predict_cost_view(request):
    result = None

    if request.method == "POST":
        form = MedicalForm(request.POST)

        if form.is_valid():
            data = form.cleaned_data
            result = predict_medical_cost(**data)
    else:
        form = MedicalForm()

    return render(request, "core/predictCost.html", {"form": form, "result": result})

def predict_risk_view(request):
    result = None

    if request.method == "POST":
        form = RiskPredictionForm(request.POST)

        if form.is_valid():
            data = form.cleaned_data
            result = predict_if_high_risk(**data)
    else:
        form = RiskPredictionForm()

    return render(request, "core/predictRisk.html", {"form": form, "result": result})