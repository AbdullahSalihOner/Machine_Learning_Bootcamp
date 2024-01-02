#NLP uygulması
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt  

data = pd.read_csv("NLPlabeledData.tsv", delimiter = "\t", quoting = 3) #quoting = 3 -> tırnak işaretlerini yok say
#%%
data = pd.read_table("NLPlabeledData.tsv") #delimiter = "\t" -> tsv dosyası olduğu için tab ile ayrılmıştır
#%%
#veri ön işleme
from nltk.corpus import stopwords #stopwords -> gereksiz kelimeler
import nltk #natural language toolkit
nltk.download("stopwords") #stopwords indir
liste = stopwords.words("english") #ingilizce stopwords listesi
liste1 = stopwords.words("turkish") #turkçe stopwords listesi
#%%
#veri temizleme işlemleri 1
ornek_metin = data.review[0] # 0. indeksteki yorumu al

#%%
#veri temizleme işlemleri 2
from bs4 import BeautifulSoup #html taglarını temizlemek için
ornek_metin1 = BeautifulSoup(ornek_metin,features="lxml").get_text() #html taglarını temizle
#%%
#veri temizleme işlemleri 3
import re #regular expression -> düzenli ifadeler için kullanılır 
ornek_metin2 = re.sub("[^a-zA-Z]"," ",ornek_metin1) #sadece a-z ve A-Z harfleri kalacak şekilde temizle 

#%%
#veri temizleme işlemleri 4
ornek_metin3 = ornek_metin2.lower() #bütün harfleri küçült  

#%%
#veri temizleme işlemleri 5
ornek_metin4 = ornek_metin3.split() #kelimeleri ayır

#%%
#veri temizleme işlemleri 6
swords = set(stopwords.words("english")) #ingilizce stopwords listesi
prnek_metin5 = [word for word in ornek_metin4 if not word in swords] #stopwordsleri temizle
#%%

#işlem fonksiyonu
def islem(review):
    review = BeautifulSoup(review,features="lxml").get_text() #html taglarını temizle 
    review = re.sub("[^a-zA-Z]"," ",review) #sadece a-z ve A-Z harfleri kalacak şekilde temizle 
    review = review.lower() #bütün harfleri küçült  
    review = review.split() #kelimeleri ayır    
    swords = set(stopwords.words("english")) #ingilizce stopwords listesi   
    review = [word for word in review if not word in swords] #stopwordsleri temizle
    return (" ".join(review)) #kelimeleri birleştir

#%%
# 1000 yorumu temizle
train_x_tum = []
for i in range(len(data["review"])): #bütün yorumları temizle
    if(i+1)%1000 == 0: #her 1000 yorumda bir ekrana yaz
        print(str(i+1) + " yorum temizlendi")
    train_x_tum.append(islem(data["review"][i])) #temizlenmiş yorumları listeye ekle
    
#%%
# veriyi train ve test olarak ayırma
from sklearn.model_selection import train_test_split #veriyi train ve test olarak ayırma    
x = train_x_tum #x -> yorumlar
y = np.array(data["sentiment"]) #y -> yorumların duyguları  0 -> negatif 1 -> pozitif 
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42) #veriyi train ve test olarak ayırma  

#%%
from sklearn.feature_extraction.text import CountVectorizer #kelimeleri vektöre çevirme 
cv = CountVectorizer(max_features = 20000) #en çok kullanılan 20000 kelimeyi al 
xtrain = cv.fit_transform(x_train) #x_train'i vektöre çevir
xtrain = xtrain.toarray() #xtrain'i array'e çevir

xtest = cv.transform(x_test) #x_test'i vektöre çevir
xtest = xtest.toarray() #xtest'i array'e çevir

#%%
"""

#model oluşturma
from sklearn.naive_bayes import GaussianNB #naive bayes algoritması
gnb = GaussianNB() #model oluştur
gnb.fit(xtrain,y_train) #modeli eğit    

"""
#%%
"""
#modeli test etme
from sklearn.metrics import roc_auc_score #roc eğrisi
y_pred = gnb.predict(xtest) #modeli test et 
accuracy = roc_auc_score(y_test,y_pred) #doğruluk oranı
print("Accuracy: %.5f%%" % (accuracy * 100.0)) #doğruluk oranı 
 
"""
#%%
#model oluşturma
from sklearn.ensemble import RandomForestClassifier #random forest algoritması  
rfc = RandomForestClassifier(n_estimators = 100) #model oluştur
rfc.fit(xtrain,y_train) #modeli eğit
xtest = cv.transform(x_test) #x_test'i vektöre çevir

xtest = xtest.toarray() #xtest'i array'e çevir
print(xtest.shape)

#%%
#modeli test etme
from sklearn.metrics import roc_auc_score #roc eğrisi
y_pred = rfc.predict(xtest) #modeli test et
accuracy = roc_auc_score(y_test,y_pred) #doğruluk oranı
print("Accuracy: %.5f%%" % (accuracy * 100.0)) #doğruluk oranı

#%%

manuel_cümle = "The movie was a huge disappointment with its predictable plot, poor acting, and dull cinematography; it completely failed to capture the essence of the original book."

# Ön işleme fonksiyonunu kullanarak cümleyi işle
manuel_cümle_islenmis = islem(manuel_cümle)

# CountVectorizer ile vektörleştirme
manuel_cümle_vektör = cv.transform([manuel_cümle_islenmis])

# Rondom Forest ile tahmin etme
tahmin = rfc.predict(manuel_cümle_vektör)

# Tahmini yazdırma
if tahmin[0] == 0:
    print("Negatif")
else:
    print("Pozitif")
    
#%%
from sklearn.metrics import accuracy_score  

accuracy = accuracy_score(y_test,y_pred) #doğruluk oranı
print("Accuracy: %.5f%%" % (accuracy * 100.0)) #doğruluk oranı

#%%

# Function to predict sentiment of a given sentence
def predict_sentiment(sentence):
    # Clean and preprocess the sentence
    sentence_processed = islem(sentence)
    # Transform the sentence to the same format as the training data
    sentence_vectorized = cv.transform([sentence_processed]).toarray()
    # Predict using the trained RandomForestClassifier model
    prediction = rfc.predict(sentence_vectorized)
    # Return the prediction
    return "Positive" if prediction[0] == 1 else "Negative"

# Asking for user input
user_input_sentence = input("Enter a sentence to determine its sentiment (Positive/Negative): ")

# Predicting the sentiment of the user input
predicted_sentiment = predict_sentiment(user_input_sentence)
print(f"The sentiment of the entered sentence is: {predicted_sentiment}")
