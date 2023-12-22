import numpy as np
import PIL.Image as img
import os
import pandas as pd
#%%
# Dosya yollarını değişkenlere atadık.
covidli = "COVID/"
covidsiz = "non-COVID/"
test = "deneme/test.png"
#%%
def dosya(yol):
    return [os.path.join(yol, f) for f in os.listdir(yol)]
# covidli ve covidsiz klasörlerindeki dosyaları listeye atadık.

#%%
def veri_donustur(klasor_adi,sinif_adi):
    goruntuler = dosya(klasor_adi)  
    goruntu_sinif = []
    
    for goruntu in goruntuler:
        goruntu_oku = img.open(goruntu).convert('L')    
        goruntu_boyutlandirma = goruntu_oku.resize((28,28))
        goruntu_normallestirme = np.array(goruntu_boyutlandirma)/255.0 
        goruntu_donusturme = goruntu_normallestirme.flatten()
        
        if sinif_adi == "covidli":
            veriler = np.append(goruntu_donusturme,[0])
        elif sinif_adi == "covid_olmayan":
            veriler = np.append(goruntu_donusturme,[1]) 
        else:
            continue
        
        goruntu_sinif.append(veriler)
    return goruntu_sinif
#%%

covidli_veri = veri_donustur(covidli,"covidli")
covidli_df = pd.DataFrame(covidli_veri)
covidli_olmayan_veri = veri_donustur(covidsiz,"covid_olmayan")
covidli_olmayan_df = pd.DataFrame(covidli_olmayan_veri) 

tum_veri = pd.concat([covidli_df,covidli_olmayan_df])

#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 

giris = np.array(tum_veri)[:,:784]
cikis = np.array(tum_veri)[:,784]
#%%
giristrain, giristest, cikistrain, cikistest = train_test_split(giris, cikis, test_size=0.2, random_state=32)
model = DecisionTreeClassifier(max_depth=5)
clf = model.fit(giristrain,cikistrain)
cikis_tahmin = clf.predict(giristest)
print("Accuracy:",metrics.accuracy_score(cikistest, cikis_tahmin))  
#%%

goruntu_oku1 = img.open(test).convert('L')  
goruntu_boyutlandirma1 = goruntu_oku1.resize((28,28))
goruntu_normallestirme1 = np.array(goruntu_boyutlandirma1)/255.0
goruntu_donusturme1 = goruntu_normallestirme1.flatten()
goruntu_donusturme1 = goruntu_donusturme1.reshape(1,-1)

print("Tahmin:",clf.predict(goruntu_donusturme1))

if clf.predict(goruntu_donusturme1) == 0:
    print("covidli")
    metin = "covidli"
if clf.predict(goruntu_donusturme1) == 1:
    print("covidsiz")
    metin = "covidsiz"    

# %%

import cv2
resim = cv2.imread(test)    
cv2.putText(resim, metin, (10,235), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),2)
cv2.imshow("tahmin", resim)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(cikistest, cikis_tahmin)  
print(cm)