from cgitb import grey
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import PIL.ImageOps 
import os,ssl,time

x=np.load('image.npz')['arr_0']
y=pd.read_csv('data.csv')['labels']

print(pd.Series(y).value_counts())

classes=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses=len(classes)

samples_per_class=5
figure=plt.figure(figsize=(nclasses*2,(1+samples_per_class*2)))
idx_cls=0
for cls in classes:
  idxs=np.flatnonzero(y==cls)
  idxs=np.random.choice(idxs,samples_per_class,replace=False)
  i=0
  for idx in idxs:
    plt_idx=i*nclasses+idx_cls+1
    p=plt.subplot(samples_per_class,nclasses,plt_idx)
    p=sns.heatmap(np.reshape(x[idx],(22,30)),cmap=plt.cm.gray,xticklabels=True,yticklabels=True,cbar=False)
    p=plt.axis('off')
    i+=1
  idx_cls+=1

  x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=9,train_size=7500,test_size=2500)
x_train_scaled=x_train/255.0
x_test_scaled=x_test/255.0
clf=LogisticRegression(solver='saga',multi_class='multinomial').fit(x_train_scaled,y_train)
y_pred=clf.predict(x_test_scaled)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

cap=cv2.VideoCapture(0)
while(True):
  try:
    ret,frame=cap.read()
    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    height,width=grey.shape()
    upper_left=(int(width/2-56),int(height/2-56))
    bottom_right=(int(width/2+56),int(height/2+56))
    cv2.rectangle(grey,upper_left,bottom_right,(0,255,0),2)
    roi=grey[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
    im_PIL=Image.fromarray(roi)
    image_bw=im_PIL.convert('L')
    image_bw_resized=image_bw.resize((28,28),Image.ANTIALIAS)
    image_bw_resized_inverted=PIL.ImageOps.invert(image_bw_resized)
    pixel_filter=20
    min_pixel=np.percentile(image_bw_resized_inverted,pixel_filter)
    image_bw_resized_invered_scaled=np.flip(image_bw_resized_inverted-min_pixel),0,255
    max_pixel=np.max(image_bw_resized_inverted)
    image_bw_resized_invered_scaled=np.array(image_bw_resized_invered_scaled/max_pixel)
    test_sample=np.array(image_bw_resized_invered_scaled).reshape(1,784)
    test_pred=clf.predict(test_sample)
    cv2.imshow('frame',grey)
    
    if cv2.waitkey(1)& 0xFF==ord("Q"):
      break
  except Exception as e:
    pass

cap.release()
cv2.destroyAllWindows()
