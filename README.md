# JNHA-CNN
CNN model for herbal component classification

############ Basic Intrsuctions ##############################

1. In the preprocess/ directory, ensure the following files are present

	'ATR FTIR jinhuacha testing selected mixture-1.csv' (a raw data file )
	preprocess_data.py

Edit the preprocess_data.py to read in the raw csv file for preprocessing. If it is necessary, you may need to customize or modify the code preprocess_data.py to produce correct preprocessing of the csv data. 

Execute 
	python preprocess_data.py
in the presence of the csv file. The output would be (i) a few png files displaying the 1D FTIR fingerprints of the herb classes, (ii) label.npy, and (iii) config.npy


2. cd ../
	Ensure the following scripts are present in the current folder
	JNHA-CNN.py
	CNN_SMOTE.py
	mycnn_model.py
	test.py

3. Ensure the *.npy data obtained from the precessing in step 1 are present in the same directory and labelled as follows
	"label.npy"
	"config.npy"
where label.npy file is the labels for the data (ie. target class) and the config.npy file is the data. 

4. Run the main.py script

5. Change parameters in the main.py script based on needs


############# Job of each script ###########################

main.py: script runs all the scripts required for the creation of the model. It also creates all the prerequisite folders and files required by the rest of the scripts.

CNN-SMOTE.py: takes the label.npy and config.npy files to artifically augment or synthesize more data. It splits its results into testing training and validation datasets. The final result is 6 npy files, 3 data files and 3 label files. They are named as 
	"Xa.npy" , "Ya.npy" for training
	"Xb.npy" , "Yb.npy" for testing
	"Xc.npy" , "Yc.npy" for validation
The files starting with "X" denote data and files starting with "Y" denote label files

mycnn_model: Creates the Convoluntional Neural Network(CNN) Model for the herb part classification. It also saves the learning curve of the model training in the "learning_curves" folder. The input is the training and testing dataset, the output given is the CNN model labelled as "model.h5" and learning curve jpeg.

test.py: Tests the saved model's accuracy and saves results in the "performance_run.csv" along with the confusion matrix plot in the "plots" folder. The input is the model and validation dataset (ie. Xc.npy & Yc.npy).The output is vales stored in the csv and a jpeg file in the plots folder.

#######################################
