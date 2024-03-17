import nltk
import shutil

source_file = open('stopwords/stopwords_ua.txt', 'rb')
path = nltk.data.path[0].split('/')[0] + '\\AppData\\Roaming\\nltk_data\\corpora\\stopwords\\ukrainian'
destination_file = open(path, 'wb+')
shutil.copyfileobj(source_file, destination_file)
