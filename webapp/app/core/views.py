from django.shortcuts import render
from .forms import MedicalForm
from .ml_model import predict_medical_cost

def home(request):
    return render(request, "core/index.html")

def predict_view(request):
    result = None

    if request.method == "POST":
        form = MedicalForm(request.POST)

        if form.is_valid():
            data = form.cleaned_data
            result = predict_medical_cost(**data)
    else:
        form = MedicalForm()

    return render(request, "core/predict.html", {"form": form, "result": result})