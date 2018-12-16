# Machine_Learning_Classification
Classification and Predictive Modeling using Newsgroup, Census, and Bank Datasets

Assignment 2: Classification and Predictive Modeling
Depaul University 
DSC 478 Programming Machine Learning Applications
Professor Aleksandar Velkoski
Fall 2018

You will experiment with various Classification Models using subsets of real-world
data.

1.	K-Nearest-Neighbor (KNN) Classification Using Newsgroups Dataset
	-	This problem uses a subset of the 20 Newsgroup dataset
	-	The full set contains 20k newsgroup documents, partitioned
		evenly across 20 different newsgroups, and has been often 
		used for experiments in text applications of machine learning
		techniques, such as classification and text clustering.
	-	The assigment dataset contains a subset of 1000 documents and
		a vocabulary of terms. Each document belongs to one of two classes,
		Hockey (label 1) or Microsoft Windows (label 0).

	a.	Create your own KNN classifier function
		-	Allow as input the training data matrix, training labels,
			instance to be classified, value of K
		-	Return predicted class for the instance and the top K 
			neighbors

	b.	Create a function to compute the classification accuracy over
		the test data set 
		-	This function will call the classifier function in part
			a on all the test instances and in each case compares the
			actual test class label to the predicted class label

	c.	Run your accuracy function on a range of values for K in order to
		compare accuracy values for different number of neighbors.
		-	Do this using Euclidean Distance as well as Cosine Similarity

	d.	Using Python, modify the training and test data sets so that term
		weights are converted to TFxIDF weights instead of raw term
		frequencies.
		-	Then, rerun your evaluation on the range of K values and
			compare the results to the results without using TFxIDF

2.	Classification Using SciKit-Learn
	-	For this problem you will experiment with various classifiers
		provided as part of the sklearn machine learning module, as well
		as with some of its preprocessing and model evaluation capabilities
	-	This problem uses a subset of a real dataset of customers for a bank.
		*Note: the bank dataset can be found in the 
		Machine_Learning_Preprocessing_and_Basic_Analysis repository.*

	a.	Load and preprocess the data using NumPy or Pandas and the 
		preprocessing functions from sklearn.
		-	Separate the target attribute (PEP) from the portion
			of the data to be used for training and testing
		-	Convert the selected dataset into the Standard Spreadsheet
			Format (sklearn functions generally assume that all attributes
			are in numeric form)
		-	Split the transformed data into training and test sets (80-20 split)

	b.	Run sklearn's KNN classifier on the test set.
		-	First normalize data so that all attributes are in the same
			scale.
		-	Generate the confusion matrix, and visualize it using
			Matplotlib
		-	Generate the classification report
		-	Compute the average accuracy score
		-	Experiment with different values of K and the weight parameter
			to see if you can improve accuracy

	c.	Repeat the classification using sklearn's decision tree classifier
		(using the default parameters), and the Naive Bayes (Gaussian)
		classifier. 
		-	For each model, compare the average accuracy scores on the
			test and training data sets.
		-	What does the comparison tell you in terms of bias-variance
			tradeoff?

3.	Data Analysis and Predictive Modeling on Census Data (adult-modified.csv)

	a.	Preprocessing and data analysis:
		-	Examine the data for missing values.
		-	Remove instances of missing values for categorical attributes
		-	For numeric attributes, impute and fill in the missing values
			using the attribute mean
		-	Examine the characteristics of the attributes, including relevant statistics
		-	Create histograms illustrating the distributions of numeric attributes
		-	Create bar graphs showing value counts for categorical attributes
		-	Perform the following cross-tabulations:
			-	Education + race
			-	Work-class + income
			-	Work-class + race
			-	Race + income

	b.	Predictive Modeling and Model Evaluation
		-	Using Pandas or sklearn, create dummy variables for
			all categorical attributes
		-	Separate the target attribute (income>50k) from the
			attributes used for taining
		-	Use sklearn to build classifiers using Naive Bayes (Gaussian),
			Decision Tree (using entropy as selection criteria), and 
			Linear Discriminant Analysis (LDA).
		-	For each of the calssifiers perform 10-fold cross-validation
			and report overall average accuracy
