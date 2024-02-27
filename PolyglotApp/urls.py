from django.contrib import admin
from django.urls import path, include
from . import views
from rest_framework import routers

router = routers.DefaultRouter()

urlpatterns = [
    path('', views.index, name="index"),
    path('predict', views.predict, name="predict"),
    path('get-accuracy', views.getAccuracy, name="getAccuracy")
]