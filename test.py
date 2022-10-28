from keras import models 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from JNHA_CNN import csvfilename

############LOAD DATASET FORM FILE##################
Y_test=np.load('Yc.npy',allow_pickle=True)
X_test=np.load('Xc.npy',allow_pickle=True)
#Y_test=np.load('label.npy',allow_pickle=True)
#X_test=np.load('config.npy',allow_pickle=True)

#####ENCODE LOADED FILE TO WORK WITH NN OUTPUT#######
encoder = LabelEncoder()
encoder.fit(Y_test)
encoded_Y = encoder.transform(Y_test)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
##########CALCULATE ACCURACY OF MODEL##################
models = models.load_model("model.h5")
y_test = models.predict(X_test)
yhat = np.argmax(y_test, axis=-1).astype('int')
acc = accuracy_score(encoded_Y, yhat)*100
print('Accuracy: %.3f' % acc)
models.evaluate(X_test,dummy_y)

print('Confusion Matrix')
print(confusion_matrix(encoded_Y, yhat))

print('Classification Report')
target_names = ['FLOWERS', 'LEAF', 'MIX','SEED']
print(classification_report(encoded_Y, yhat, target_names=target_names))
ahat=np.argmax(dummy_y, axis=-1).astype('int')
print('The final accuracy for the model is %.3f' % acc)

##########CALCULATE ACCURACY PER CLASS##################
cm = confusion_matrix(encoded_Y, yhat)
accm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
acclist=accm.diagonal()
acclist= [a*100 for a in acclist]
acclist.append(acc)

#################SAVE OUTPUT IN FILE#####################
accarray=np.array(acclist)
with open(csvfilename,'a') as csvfile:
    np.savetxt(csvfile,accarray[np.newaxis],delimiter=',',fmt='%s')

with open('CM_4MORERUN.csv','a') as csvfile:
    np.savetxt(csvfile,confusion_matrix(encoded_Y, yhat),delimiter=',',fmt='%s')

file=open('REPORT_4MORERUN.txt','a')
target_names = ['Flower', 'Leaf', 'Mix','Seed']
file.write('Confusion Matrix\n'+
           str(confusion_matrix(encoded_Y, yhat))+'\n'+
           '\nClassification Report \n'+
           str(classification_report(encoded_Y, yhat, target_names=target_names))+'\n') 
file.close()



##########confusion matrix plot##################
cm=confusion_matrix(encoded_Y, yhat)
labels = ["{0:0.0f}".format(value) for value in
                cm.flatten()]
labels = np.asarray(labels).reshape(4,4)
ax = sns.heatmap(cm, annot=labels, fmt='', cmap='viridis',cbar=False)
ax.set_title('Herb part classification Confusion Matrix \n');
ax.set_xlabel('\nPredicted Part Classification')
ax.set_ylabel('Actual Part Classification ');
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(target_names)
ax.yaxis.set_ticklabels(target_names)
plt.rcParams["figure.figsize"] = (9,7)
run_no=(len(os.listdir("plots\\")))+1
path=os.path.join(os.getcwd(),'plots','confusion_matrix_plot#'+str(run_no) +'.png')
plt.savefig(path,dpi=300,bbox_inches='tight')









