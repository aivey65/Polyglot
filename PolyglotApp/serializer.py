from rest_framework import serializers 
from .models import DataPoint 

class DataPointSerializers(serializers.ModelSerializer): 
    class meta: 
        model=DataPoint 
        fields='__all__'