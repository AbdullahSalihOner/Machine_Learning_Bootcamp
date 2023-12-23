import pickle
with open("scaler.pkl","rb") as file:
    load_sc=pickle.load(file)
#%%
with open("model.pkl","rb") as file:
    load_model = pickle.load(file)
#%%
import warnings
warnings.filterwarnings('ignore')
#%%
import numpy as np
new_predict = load_model.predict(load_sc.transform(np.array([[1,148,72,20,1,33.6,0.427,20]])))
print(new_predict)
#%%
from flask import Flask,request,render_template 
app = Flask(__name__)
@app.route("/" ,methods=["GET","POST"])

def mltahmin():
    tahmin = None
    tahmin1 =None
    if request.method=="POST":
        Pregnancies=float(request.form["Pregnancies"])
        Glucose=float(request.form["Glucose"])
        BloodPressure=float(request.form["BloodPressure"])
        SkinThickness=float(request.form["SkinThickness"])
        Insulin=float(request.form["Insulin"])
        BMI=float(request.form["BMI"])
        DiabetesPedigreeFunction=float(request.form["DiabetesPedigreeFunction"])
        Age=float(request.form["Age"])
        
        kullanici_verisi = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        tahmin1 = load_model.predict(load_sc.transform(kullanici_verisi))
    return render_template("web.html", tahmin=tahmin1)

if __name__=="__main__":
    app.run(port=5001)
