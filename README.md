# Polyglot (WIP)
A language detection machine learning model

## About
This language detection model can detect 42 different languages with ~96% accuracy. It is still a work in progress.

## Corpus Source
The data used to train the model is a subset of this data set: https://www.kaggle.com/datasets/chazzer/big-language-detection-dataset?resource=download
The original source contains around 10,000,000 rows of language data in over 400 different languages.

There are other, smaller data sets on Kaggle and elsewhere on the internet, but I wanted to build a model with more diverse language representation. For example, another language detection set with the more downloads on Kaggle only features 17 different languages. As I develop this project, I plan to find a balance between accuracy and number of languages. 

The full list of the languages that can be detected so far are as follows:
> English •
Mandarin Chinese •
Hindi •
Spanish •
French •
Arabic •
Bengali •
Russian •
Portuguese •
Indonesian •
Urdu •
German •
Japanese •
Swahili •
Punjabi •
Tamil •
Korean •
Vietnamese •
Javanese •
Italian •
Thai •
Tagalog •
Polish •
Yoruba •
Ukrainian •
Igbo •
Nepali •
Romanian •
Dutch •
Malay •
Afrikaans •
Greek •
Swedish •
Hebrew •
Latin •
Sanskrit •
Irish •
Maori •
Cherokee •
Navajo •
Hawaiian •
Samoan

## Technologies and Libraries
I used the **scikit-learn** CountVectorizer and the MultinomialNB classifying model.

## Development Workflow
### Dependencies
Dependencies used are listed in requirements.txt. You can create a virtual environment and then run 

> pip install -r requirements.txt

to install them all at once.

### Creating a Django Project and Running a Development Server
For reference, this is how you create a new Django project. Specifying the location of the project (".") will create a project inside the current directory, which is useful if you already have a github repository.

> django-admin startproject [project-name] [path]

To create a Django application (located inside a Django project):

> django-admin startapp [app-name]

To start the development server, use this command:

> python3 manage.py runserver

To create a database using Django:

> python manage.py makemigrations

> python manage.py sqlmigrate [name-of-app] [migration-initial-number]

> python manage.py migrate

To create an admin account:

> python manage.py createsuperuser

#### Notes on Deploying
Instructions:
Coming Soon...

Here are some things to remember when deploying this project:
1. Remember to change the debug setting in "Polyglot/settings.py" to "False".
