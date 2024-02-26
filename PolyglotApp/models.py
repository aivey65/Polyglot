from django.db import models


class DataPoint(models.Model):
    sentence = models.TextField()
    lanCode = models.CharField(max_length=3)

    def __str__(self):
        return self.lanCode + ", " + self.sentence