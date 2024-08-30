# Impact of unsupervised feature selection techniques
This repository contains the code used in the article titled "The Impact of Unsupervised Feature Selection Techniques on the Performance and Interpretation of Defect Prediction Models."


## Prerequisites
Because some of the unsupervised feature selection algorithms used are implemented using matlab, this experiment uses these techniques by launching the Matlab engine using the 'matlab.engine' module for python. Therefore, it is necessary to install a version of the matlab product that is compatible with python and set the path in matlab.
## Repository Structure

* code
	* run.py: The main approach to running the entire python code
	*  FeatureSelection: The various feature selection methods used are called from the methods in this folder.
	* FeatureSelectionAlgorithms：This folder holds the specific implementation of each feature selection algorithm. The feature selection algorithm we are using originates from: <https://github.com/farhadabedinzadeh/AutoUFSTool>,<https://github.com/csliangdu/FSASL>,<https://jundongl.github.io/scikit-feature/>
	* ModelConstruction
	  * partition_dataset.py: Divide the dataset by invoking out-of-sample bootstrap validation techniques.
	  * fast_bootstrapCV.py: Realization of out-of-sample bootstrap validation techniques.
	  * SC.py: Implementation of a classifier based on spectral clustering.
	  * TCL.py: TCL unsupervised classifier.
	  * CLA.py: CLA unsupervised classifier.
	  * xgb_rfclassifiers.py: Wrapper supervised feature selection of supervised classifiers used in XGBF and RFF.
	  * ConstructModel.py: Supervised and unsupervised models were constructed using logistic regression, random forest, spectral clustering based and k-means based classifiers and the model performance was evaluated using evaluation metrics AUC,MCC,IFA,Recall@20%.
	* ModelEvalution
	  *  classificationMeasure.py: Non-effort-aware evaluation measures
	  * rankMeasure.py: Effort-aware evaluation measures

* results: Deposit of research results.
