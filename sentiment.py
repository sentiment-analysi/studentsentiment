# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
nltk.download('stopwords')

loaded_model=pickle.load(open('D:/work/trained.sav','rb'))
cv1=pickle.load(open('D:/work/count-Vectorizer.pkl','rb'))

def predict_sentiment1(sample_review):
  sample_review = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_review)
  sample_review = sample_review.lower()
  sample_review_words = sample_review.split()
  stop_words = set(stopwords.words('english'))
  stop_words.remove('not')
  sample_review_words = [word for word in sample_review_words if not word in stop_words]
  ps = PorterStemmer()
  final_review = [ps.stem(word) for word in sample_review_words]
  final_review = ' '.join(final_review)

  temp = cv1.transform([final_review]).toarray()
  return loaded_model.predict(temp)

sample_review = 'he was  kind'

if predict_sentiment1(sample_review):
  print('This is a POSITIVE review.')
else:
  print('This is a NEGATIVE review!')