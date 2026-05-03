from django.urls import path
from .views import home, predict_cost_view, predict_risk_view

urlpatterns = [
    path("", home, name="home"),
    path("predictCost", predict_cost_view, name="predict_cost"),
    path("predictRisk", predict_risk_view, name="predict_risk"),
]