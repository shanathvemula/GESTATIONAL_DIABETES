from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import svm
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from django.http import HttpResponse

def input(request):
    return render(request,'index.html')
def index(request):
    df = pd.read_csv("C:\\Users\\shanath\\Downloads\\pima-diabetic.csv")
    X = df.iloc[:, 0:8]
    y = df.iloc[:, 8]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=195)
    svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
    y_pred= svm_model_linear.predict(X_test)
    z=format(svm_model_linear.score(X_train, y_train),'.2f')
    p=format(svm_model_linear.score(X_test, y_test),'.2f')
    q=metrics.classification_report(y_test, y_pred, target_names=['tested=_positive', 'tested_negative'])
    r=confusion_matrix(y_test, y_pred)
    i=X_train.shape[0]
    j=X_test.shape[0]
    slices = [i,j]
    a = 'True ' + str(i) + ' rows'
    b = 'False ' + str(j) + ' rows'
    labels = [a, b]
    sizes=[a,b]
    fig1, ax1 = plt.subplots()
    ax1.pie(slices, autopct='%1.1f%%', explode=(0, 0.1), labels=labels, shadow=True, startangle=90)
    plt.title('pie chart')
    plt.legend(title="train data", loc="lower right")
    ax1.axis('equal')
    plt.tight_layout()
    plt.savefig('app/static/img/pie.png')
    preg=request.GET['preg']
    a=float(preg)
    plas = request.GET['plas']
    b=float(plas)
    pres = request.GET['pres']
    c=float(pres)
    skin = request.GET['skin']
    d=float(skin)
    insu = request.GET['insu']
    e=float(insu)
    mass = request.GET['mass']
    f=float(mass)
    pedi = request.GET['pedi']
    g=float(pedi)
    age = request.GET['age']
    h=float(age)
    dataClass = svm_model_linear.predict([[a,b,c,d,e,f,g,h]])
    if dataClass == 0:
        return HttpResponse("Negative<br>Accuracy on training set:"+str(z)+"<br>Accuracy on test set:"+str(p)+"<br>"+q+"<br>confusion matrix"+str(r))
    else:
        return HttpResponse("positive<br>Accuracy on training set:"+str(z)+"<br>Accuracy on test set:"+str(p)+"<br>"+q+"<br>confusion matrix"+str(r))