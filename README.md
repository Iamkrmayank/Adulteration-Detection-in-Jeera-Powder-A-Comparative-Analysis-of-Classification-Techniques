# Adulteration Detection in Jeera Powder: A Comparative Analysis of Classification Techniques Using Wooden Dust as an Adulterant

This is a research study that aims to detect and classify adulteration (contamination) in jeera powder (a common spice) using wooden dust as a contaminant.

# Goal:
To develop an effective method for detecting adulteration in jeera powder, ensuring food safety and quality control.

# 1.Dataset
The dataset used in this study pertains to the analysis of jeera powder (cumin powder) images. Initially, macro images of jeera powder were captured to extract relevant statistical features. These features were then compiled into a structured dataset. The objective of this dataset is to utilize the extracted features for classification tasks, potentially to distinguish between different qualities or types of jeera powder.

Data Collection
The images of jeera powder were obtained using a high-resolution camera capable of capturing macro images. This ensured detailed and high-quality images, which are essential for accurate feature extraction. The images were processed using image analysis techniques to derive several statistical features.
Data Extraction
Data extraction refers to the process of retrieving specific data from various sources to transform it into a format suitable for further processing and analysis. This involves identifying, retrieving, and transforming data so that it can be used for various purposes such as analysis, reporting, and integration into databases or applications.

Dataset Features
The dataset comprises the following columns, each representing a specific statistical attribute extracted from the jeera powder images:

mean_intensity: The average intensity of the pixels in the image. This feature indicates the overall brightness of the image.
median_intensity: The median value of the pixel intensities in the image. This is useful for understanding the central tendency of the pixel values, particularly in the presence of outliers.
mode_intensity: The most frequently occurring pixel intensity value in the image. This feature can highlight the most common pixel value, giving insights into the most dominant intensity level.
std_intensity: The standard deviation of the pixel intensities. This measure indicates the spread or variability of the intensity values around the mean, reflecting the contrast in the image.
var_intensity: The variance of the pixel intensities. Similar to standard deviation, this feature represents the degree of dispersion in the intensity values.
skewness: A measure of the asymmetry of the distribution of pixel intensities. This can help in understanding the distribution shape and whether there are more dark or light pixels.
kurtosis: A measure of the "tailedness" of the distribution of pixel intensities. High kurtosis indicates more pixels with extreme intensity values, either very dark or very bright.
Outcome: The target variable, which contains binary values (0 or 1). This column indicates the classification result based on the extracted features. For instance, it may denote different quality grades of the jeera powder.

Class Imbalance and SMOTE Application
Initially, the 'Outcome' column exhibited a significant class imbalance with the following distribution:

Class 0: 25 instances
Class 1: 225 instances
To address this imbalance and ensure that the machine learning models are not biased towards the majority class, we applied the Synthetic Minority Over-sampling Technique (SMOTE). SMOTE generates synthetic samples for the minority class (Class 0 in this case) by interpolating between existing samples. This helps to create a more balanced dataset without merely duplicating existing records.

Post-SMOTE Application
After applying SMOTE, the dataset achieved a balanced distribution of the 'Outcome' values, resulting in equal representation of both classes. This balanced dataset is crucial for training robust and unbiased models, improving their ability to generalize and perform well on unseen data.

Class 0: 225 instances
Class 1: 225 instances

# Link for Dataset: https://github.com/Iamkrmayank/Adulteration_Detection/blob/main/Jeera_New.xlsx

2.Data Preparation
To ensure the quality and reliability of the dataset, several preprocessing steps were undertaken:
(a). Data Exploration 
This involves understanding the dataset by summarizing its main characteristics often using visual methods. It includes inspecting the data types, checking for missing values, and computing basic statistics.
•  Checking data types and structure: df.info()
•  Summarizing statistics: df.describe()
(b). Data Preprocessing
This involves cleaning and transforming raw data into a format that can be used for modelling. It includes handling missing values, encoding categorical variables, and feature extraction or creation.
●	Handling Missing Values by Removal of Incomplete Records: df.dropna()

(c)  Data Scaling:.
This involves standardizing or normalizing the features so that they have similar scales, which is important for many machine learning algorithms.
•  Standard scaling (zero mean, unit variance): StandardScaler()
•  Min-max scaling (scaling features to a fixed range, e.g., 0 to 1): MinMaxScaler()

# Implementation Flow:

![Workflow](https://github.com/user-attachments/assets/e7d45d17-ca76-4b60-8444-3aef1e85804c)

# Stacking Classifier 

![image](https://github.com/user-attachments/assets/d6c50dfb-9443-49d0-9fa4-e23713c2c514)

An ensemble learning technique called stacking classifier leverages the strengths of multiple base models to improve prediction accuracy. It works by combining predictions from many base models, each trained with different hyperparameters or techniques, to create a robust meta model. 
Using the base models' diverse viewpoints on the dataset, this meta model—also called a blender or aggregator—skillfully learns to synchronise the predictions produced by the base models. 
Stacking improves predictive performance and offers a flexible approach to complex categorization tasks by leveraging the diversity among base models.

A stacking classifier is an ensemble method that combines multiple machine learning models to improve predictive performance

1. Training Base Models (QDA, MLP Classifier, AdaBoost)
	QDA (Quadratic Discriminant Analysis)
	MLP Classifier (Multi-Layer Perceptron)
	AdaBoost (Adaptive Boosting)
  These base models are trained on the same training set independently. Mathematically, if 80% is the training set and 20% is the target variable:
  QDA:         y_QDA =QDA(x)
  MLP CLassifier:  y_MLP =MLP(x)
  AdaBoost∶ y_Ada =AdaBoost(x)
2. Creating a New Training Set
The predictions from the base models are used to create a new training set. This new training set is typically composed of the outputs (predictions) of the base models. If there are N training samples and k base models, the new training set 0.80 will have 8 features (each feature being the prediction of a base model)

3. Training the Meta Model (Logistic Regression)
The new training set 0.80 is used to train a meta-model (in this case, Logistic Regression). The meta-model learns how to combine the predictions of the base models to make the final prediction. Mathematically:
Meta Model: y =  LogisticRegression(x)

5. Making Final Predictions
The final predictions are made by the meta-model. The new data (test set) is passed through the base models to get their predictions, which are then used as input for the meta-model to produce the final predictions. If X_test is the test set:
(a) Get predictions from base models
(b) Form the new test set for the meta-model
(c) Make final predictions with the meta-model:
Meta Model: y =  LogisticRegression(x)

# Logistic Regression
Logistic regression models the probability that a given input X belongs to a certain class y. For binary classification, y can take on values 0 or 1. Instead of modelling y directly, logistic regression models the probability that y = 1 using the logistic function (also called the sigmoid function).
Logistic Function (Sigmoid Function)
The logistic function is defined as:
σ(z)=1/(1+ e^(-z) )
where z is a linear combination of the input features. The logistic function maps any real-valued number into the range [0, 1], making it suitable for probability estimation.

Logistic Regression Model
In logistic regression, the probability that y = 1given the input features x = {x1,x2,x3,}
P(x)=σ(z)=1/(1+e^z )

where
	z= W^T x+b= w_1 x_1+ w_2 x_2+ w_3 x_3+⋯+ w_n X_n+b 
	w is the vector of weights (coefficients).
  b is the bias term (intercept).
  x is the input feature vector.

# GaussianNB
Gaussian Naive Bayes (GaussianNB) is a classification algorithm that applies the principles of Bayes' theorem with the assumption of independence among predictors, and it
assumes that the continuous features follow a Gaussian (normal) distribution. Here's an explanation of Gaussian Naive Bayes with relevant equations:

# Bayes' Theorem
Bayes' theorem provides a way to calculate the posterior probability P(X)  
from the prior probability P(y) , the likelihood P(X|y) , and the evidence P(X) 

# Naive Bayes Assumption
The "naive" assumption in Naive Bayes is that all features X_i  are independent given the class y

# Gaussian Naive Bayes
In Gaussian Naive Bayes, we assume that the continuous features follow a Gaussian distribution. The probability density function of the Gaussian distribution is given

# XGBoost
XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. 
It implements machine learning algorithms under the Gradient Boosting framework.

# Overview of Gradient Boosting
Gradient Boosting is a technique where new models are trained to correct the errors made by previous models. The key idea is to build the model in a stage-wise fashion, minimizing the loss function by adding weak learners (usually decision trees) to the ensemble.
Objective Function
The objective function in XGBoost combines a loss function that measures the model's fit and a regularization term that penalizes the complexity of the model to prevent overfitting.

# Regularization Term
The regularization term Ω(f_k )  typically includes the number of leaves in the tree and the sum of the squared leaf weights, which helps control the complexity of the model

# Additive Training
In Gradient Boosting, we add one new function (tree) at a time to minimize the objective. If y_i^(^(t))   is the prediction of the ensemble at iteration t, the model is updated.

Taylor Expansion
XGBoost uses a second-order Taylor expansion to approximate the loss function for efficient optimization

# AdaBoost
AdaBoost, short for Adaptive Boosting, is an ensemble learning algorithm that combines multiple weak classifiers to create a strong classifier. It iteratively trains weak classifiers on different subsets of the training data, assigning higher weights to misclassified samples at each iteration. This focus on challenging examples allows subsequent weak classifiers to improve their performance. In each iteration, a weak classifier is trained on the data with current sample weights, aiming to minimize classification error. 
The algorithm then calculates classifier weights based on weak classifier errors, updating sample weights to prioritize misclassified examples. After normalization, the weak classifiers' predictions are combined using a weighted majority vote, with their weights considered. 
This process is repeated for a specified number of iterations, producing a final prediction by aggregating the weak classifiers' predictions. AdaBoost effectively reduces error by aligning weak models into a strong one, making it a powerful technique in machine learning.
Overview of AdaBoost
AdaBoost works by iteratively adding weak classifiers to form a strong classifier. It adjusts the weights of misclassified samples to focus more on the harder cases in subsequent iterations.

# Butterfly Optimization Algorithm (BOA)
The Butterfly Optimization Algorithm (BOA) is a nature-inspired metaheuristic optimization technique based on the foraging behavior and sensory perception of butterflies. It was introduced as a tool for solving complex optimization problems by mimicking how butterflies use their sense of smell and visual cues to search for nectar in their environment.
Key Concepts in BOA
1.	Butterflies and Sensory Modality:
○	Each butterfly represents a solution in the search space.
○	Butterflies can perceive their environment using sensory modalities like smell intensity, which correlates to the quality of the solution.
2.	Fragrance (Fitness):
○	Fragrance represents the objective or fitness value of a solution.
○	It is influenced by sensory modality and the distance between butterflies in the search space.
3.	Movement Mechanisms: BOA utilizes two modes of movement:
○	Global Search: Explores the search space broadly to avoid local optima.
○	Local Search: Focuses on intensifying the search near high-quality solutions.
4.	Parameters:
○	Population Size: Number of butterflies (solutions) in the search space.
○	Iterations: Number of optimization cycles.
○	Sensory Modality (a): Governs the influence of the fragrance.
○	Constant (c): Controls the rate of butterfly movement.
○	Probability (p): Determines the likelihood of global or local search.

5.	Algorithm Parameters:
○	Population size: 20 butterflies representing different combinations of hyperparameters.
○	Iterations: 20 iterations to explore the search space.
○	Sensory modality (a): 0.01.
○	Constant for butterfly movement (c): 0.01.
○	Probability for movement (p): 0.8 (determining local vs. global search).
6.	Hyperparameter Space:
○	Five key hyperparameters of the Gradient Boosting Classifier were optimized:
■	Number of estimators (n_estimators): Range (100, 300).
■	Maximum tree depth (max_depth): Range (3, 7).
■	Minimum samples to split a node (min_samples_split): Range (2, 4).
■	Learning rate (learning_rate): Range (0.01, 0.3).
■	Subsample ratio (subsample): Range (0.8, 1).
7.	Initialization:
○	The initial population of butterflies was randomly generated within the bounds of the hyperparameters.
8.	Fitness Evaluation:
○	The fitness function was defined as the accuracy of the Gradient Boosting Classifier on the test set.
○	Each butterfly's position (hyperparameters) was evaluated by training and testing the model, with accuracy as the objective metric.
9.	Butterfly Movement:
○	Global Search: Applied when a random probability exceeded p. Positions were updated based on a sensory modality factor (a) and random noise.
○	Local Search: Applied when the probability was less than p. Positions were adjusted relative to the best-known butterfly's position, guided by a constant (c) and random perturbations.
10.	Constraint Handling:
○	Each hyperparameter was clipped to its predefined bounds after every update to ensure valid configurations.
11.	Optimization Loop:
○	For each iteration:
■	The fitness of all butterflies was calculated.
■	The best-performing butterfly (highest accuracy) was identified and updated.
■	Butterfly positions were updated using the movement equations.
○	The process was repeated for 20 iterations to ensure convergence.
12.	Outcome:
○	The BOA identified the optimal hyperparameter configuration for the Gradient Boosting Classifier that maximized accuracy on the test set.
Explainable AI (XAI)
Explainable AI (XAI) refers to techniques and methods that make the predictions and decisions of machine learning models transparent, interpretable, and understandable to humans. The goal is to bridge the gap between the "black-box" nature of many complex models and the need for accountability, trust, and actionable insights.
Model Training
1.	Model Selection:
○	A Random Forest Classifier was chosen for its robustness and ability to handle non-linear relationships. The model was trained on the preprocessed training data (x_train, y_train).
2.	Evaluation:
○	The trained model was used to predict the target variable on the test data (x_test). Various performance metrics, including accuracy and confusion matrix, were computed to validate the classifier's performance.

Explainable AI (XAI) Techniques
To understand the model's predictions and ensure interpretability, two state-of-the-art XAI techniques were applied:
1.	SHAP (SHapley Additive exPlanations):
○	Global Interpretability:
■	The SHAP TreeExplainer was used to compute SHAP values for each feature, providing insight into the overall impact of each feature on the model's predictions.
■	A summary plot was generated to visualize the contribution of features across the entire test dataset, highlighting the most important predictors.
○	Local Interpretability:
■	For a specific prediction (e.g., the first test instance), a waterfall plot was created to explain the contribution of each feature towards the prediction for the positive class (class 1).
■	The base_value (expected value of the model) and shap_values for the instance were combined to illustrate how individual features influenced the prediction.
2.	LIME (Local Interpretable Model-agnostic Explanations):
○	The LIME Tabular Explainer was applied to the test dataset.
○	A single instance from the test set was selected, and its prediction was explained using LIME. The explanation highlighted the most significant features contributing to the model's decision.

Visualization
1.	SHAP Summary Plot:
○	A summary plot was generated to display the average magnitude of SHAP values for each feature, indicating their overall importance to the model.
2.	SHAP Waterfall Plot:
○	A waterfall plot was created for an individual prediction to show the cumulative contribution of features, starting from the base value to the final prediction.
3.	LIME Visualization:
○	LIME produced a detailed visualization in a notebook interface, illustrating the local feature contributions for the selected instance.

# Result and Discussion
In our analysis, we used various machine learning models to predict our target variable. The models evaluated and their corresponding accuracies are summarized in the table provided

![image](https://github.com/user-attachments/assets/91de862b-9545-441d-93bd-6905a82b1b24)

we achieved the highest overall accuracy using a Stacking Classifier, an advanced ensemble learning technique. In our stacking approach, we used the following base models:
●	Quadratic Discriminant Analysis (QDA)
●	MLPClassifier (Multi-layer Perceptron) with a neural network architecture of two hidden layers (100 and 50 neurons, respectively) and a maximum of 1000 iterations.
●	AdaBoostClassifier with 100 estimators.
The final estimator for stacking was Logistic Regression. This combination allowed us to leverage the strengths of different types of classifiers:
●	QDA handles data with distinct class distributions effectively.
●	MLPClassifier captures complex patterns through neural networks.
●	AdaBoost focuses on difficult-to-classify instances.
By stacking these models and using logistic regression as the final estimator, we effectively captured diverse patterns and interactions in the data, resulting in improved predictive performance. The stacking classifier outperformed all individual models, demonstrating the power of combining different machine learning techniques to achieve higher accuracy and robustness in predictions.

![image](https://github.com/user-attachments/assets/e52a6b92-22e8-46e8-a1ce-761815aa6259)

The ROC curve in our code is generated using the following steps:
1.	Train-Test Split: Split the dataset into training and testing sets.
2.	Feature Scaling: Scale the features using StandardScaler.
3.	Define Base Models: Define the base models for stacking (QDA, MLPClassifier, and AdaBoostClassifier).
4.	Define Final Estimator: Define the final estimator for stacking (Logistic Regression).
5.	Create and Train Stacking Classifier: Create and train the stacking classifier.
6.	Make Predictions: Make predictions on the test set.
7.	Compute Accuracy and Confusion Matrix: Calculate the accuracy and confusion matrix to evaluate performance.
8.	Compute ROC Curve and AUC:
o	Calculate the probabilities for the positive class.
o	Compute the false positive rate and true positive rate using roc_curve.
o	Compute the area under the curve using auc.
o	Plot the ROC curve.
ROC Curve Interpretation
●	Orange Line (ROC Curve): Represents the trade-off between TPR and FPR at different threshold values. The closer this curve follows the left-hand border and then the top border of the ROC space, the better the performance of the classifier.
●	Diagonal Line (Baseline): Represents a random classifier (AUC = 0.5). If your ROC curve is above this line, your classifier is better than random guessing.
●	AUC = 0.98: Indicates excellent performance. An AUC of 0.98 means the model has a 98% chance of distinguishing between a positive and a negative instance correctly.

BOA (Base = Gradient Boosting)  Result : 
Best Hyperparameters found: [227.87471569   3.9954015    2.97466434   0.24703248   0.95014934]

Explainable AI (XAI):
 
SHAP is a popular explainable AI (XAI) technique that helps to understand how features contribute to a model's prediction.
Here detailed explanation of the components:
Key Elements : 
1.	SHAP Values:
○	SHAP values quantify the contribution of each feature to the model's output (prediction). Positive values indicate that the feature pushes the prediction higher, while negative values indicate that it pushes the prediction lower.
2.	f(x) :
○	f(x)=0.38 : This is the predicted value for the given instance. It represents the output of the model for the specific data point being explained.
3.	Expected Value E[f(x)]:
○	This is the base value or the average model prediction across all training samples. It serves as a starting point for the SHAP explanation.
4.	Features ("Energy" and "Contrast"):
○	The two features, Energy and Contrast, are shown with their respective SHAP values, indicating how much they contributed to the deviation from the expected value to the prediction (f(x)f(x)f(x)).
■	Energy: Contributed −0.297, decreasing the prediction.
■	Contrast: Contributed −0.733, also decreasing the prediction.
5.	Feature Contributions:
○	The bottom text shows contributions in absolute terms, which are very small in this case:
■	Contrast: +0.0003
■	Energy: −0.0003
Interpretation:
●	The model's prediction (f(x)=0.38) is derived by starting from the expected value E[f(x)] and adjusting it based on the contributions of the features. The SHAP values reveal how much each feature influenced the prediction:
○	The Energy feature reduced the predicted value by 0.297.
○	The Contrast feature reduced the predicted value by 0.733.
●	Overall, both features contributed negatively to the model's prediction, resulting in the final output of f(x).






