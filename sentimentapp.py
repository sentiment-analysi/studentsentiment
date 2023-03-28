# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 19:18:41 2023

@author: EXSON
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import streamlit as st
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
        pred=loaded_model.predict(temp)
        if(pred[0]==0):
            return 'This is a NEGATIVE review'
        else:
            return 'This is a POSITIVE review'
    
    
    
    
    
def main():
    
    st.title('Student sentiment analysis')
    
    
    review1=st.text_input('How was the course experience?')
    review2=st.text_input('Tell us about the instructor?')
    review3=st.text_input('Was the material provided useful?')
    
    
    result1=''
    result2=''
    result3=''
    
    
    if st.button('predict'):
        result1= predict_sentiment1(review1)
        result2= predict_sentiment1(review2)
        result3= predict_sentiment1(review3)
      
        
    st.success(result1)
    st.success(result2)
    st.success(result3)
    
   
    

if __name__=='__main__':
    main()