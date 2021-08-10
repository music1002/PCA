from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

data= fetch_lfw_people(min_faces_per_person = 70, resize = 0.4)
X=data.data
y=data.target
target_names=data.target_names
images=data.images
n,h,w=images.shape
def plot(image,titles,h,w,rows=3,cols=3):
    plt.figure(figsize=(2*rows,2*cols))
    for j in range(rows*cols):
        plt.subplot(rows,cols,j+1)
        plt.imshow(image[j].reshape(h,w),cmap="gray")
        plt.title(target_names[titles[j]])
        plt.axis("off")
plot(X,y,h,w)
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)
pc=PCA(n_components=500)
pc.fit(X_train)
X_train_trans=pc.transform(X_train)
X_test_trans=pc.transform(X_test)

clf=MLPClassifier(hidden_layer_sizes=(512,),batch_size=128,verbose=True,early_stopping=True)
clf.fit(X_train_trans,y_train)
y_pred=clf.predict(X_test_trans)
print(classification_report(y_test, y_pred,target_names=target_names))

p=PCA()
p.fit(X_train)
#print(p.transform(X_train).shape)
s=np.sum(p.explained_variance_)
var=p.explained_variance_ #gives variances along each direction
c=p.components_
inx_sort=np.argsort(var)
inx_sort=inx_sort[::-1]

_sum=0
principal_vec=[]
principal_val=[]
i=0
while (_sum<0.98*s):
    principal_vec.append(c[inx_sort[i],:])
    principal_val.append(var[inx_sort[i]])
    _sum+=var[inx_sort[i]]
    i+=1
print("No of components:{}".format(i))
principal_vec=np.matrix(principal_vec)
print("*"*40)
    
X_train_trans=np.dot(X_train,principal_vec.T)
X_test_trans=np.dot(X_test,principal_vec.T)

clf2=MLPClassifier(hidden_layer_sizes=(512,),batch_size=128,verbose=True,early_stopping=True)
clf2.fit(X_train_trans,y_train)
print(classification_report(y_test, y_pred,target_names=target_names))
mean_imgs=[]
for x in range(i):
    vec=principal_vec[x,:]
    img=vec.reshape(h,w)
    mean_imgs.append(img)
mean_imgs=np.array(mean_imgs)
titles=[f"eigenvectors--{x}"for x in range(i)]
def plot1(image,titles,h,w,rows=3,cols=3):
    plt.figure(figsize=(2*rows,2*cols))
    for j in range(rows*cols):
        plt.subplot(rows,cols,j+1)
        plt.imshow(image[j].reshape(h,w),cmap="gray")
        plt.title(titles[j])
        plt.axis("off")
plot1(mean_imgs,titles,h,w)
#print(titles)

