from rest_framework import viewsets 
from rest_framework.decorators import api_view 
from django.core import serializers 
from rest_framework.response import Response 
from rest_framework import status 
from django.http import JsonResponse 
from rest_framework.parsers import JSONParser 
from .models import DataPoint 
from .serializer import DataPointSerializers 
from django.shortcuts import render, redirect 
from django.contrib import messages

import pickle
import json 
import pandas as pd 
import numpy as np 
import regex as re

from language_detector import createPrediction

class DataPointView(viewsets.ModelViewSet):
    queryset = DataPoint.objects.all()
    serializer_class = DataPointSerializers

@api_view(['GET', 'POST'])
def predict(request):
    try:
        prediction_text = request.POST.get("text-input", "")
        result = createPrediction(prediction_text)
        return render(request, "results.html", {"data": result})
    except ValueError as e: 
        return Response(e.args[0], status.HTTP_400_BAD_REQUEST)

def index(request):
    return render(request, "index.html")

def showPredictionForm(request):
    return render(request, "predictionForm.html")