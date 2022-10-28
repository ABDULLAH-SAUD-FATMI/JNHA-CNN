"""
JNHA CNN model
Objective: To run all the scripts required for the creation and testing of the herb classification model
"""
import runpy
import os

csvfilename='performance_4MORERUN.csv' #change file name to whatever you want to name your results
loop=10 # choose how many runs you want default is 10
plot=os.path.join(os.getcwd(),'plots') #folderpath for folder saving plots
learning_curves=os.path.join(os.getcwd(),'learning_curves')#folderpath for folder saving learning curves

#create prerequisite files and folders for run
if os.path.exists(csvfilename): # to check for existing file
    print(csvfilename+' found')
else:
    #if file not found create new csv file to save results in
    import csv
    with open(csvfilename, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['Flower_accuracy','Leaf_accuracy','Mix_accuracy','Seed_Accuracy','Total_Accuracy'])
        print(csvfilename+' created in '+os.getcwd())

if not os.path.exists(plot):
      os.makedirs(plot ) 
if not os.path.exists(learning_curves):
      os.makedirs(learning_curves) 
      
#runs all the scripts required for CNN model creation
for i in range(loop):
    runpy.run_path("CNN_SMOTE.py")    #to run data augmentation and
    runpy.run_path("mycnn_model.py")  #create the CNN model based on the augmented data
    runpy.run_path("test.py")         #to test the accuracy and save results by class