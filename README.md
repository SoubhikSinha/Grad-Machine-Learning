# Machine Learning

Hello Everyone !
<br>

Welcome to my <b>`Machine Learning`</b> repository. Here, I will talk about some graduate-level concepts, algorithms, tools and frameworks related to Machine Learning, which bolster the foundation of research and development, especially in the field of <u>[Data Science](https://www.ibm.com/topics/data-science)</u>.
<br>

> NOTE : The programming language used here (i.e., in every jupyter notebook) is **Python**, as it is the most widely used in the industry.

<br>

Let us now go over the notebooks mentioned here. <i>For your information, the <u>sequence</u> below is organized by increasing levels of complexity of the work done in each notebook.</i>
<br>
<br>

### **[Python - Data Analysis - Basics](https://github.com/sricks404/Machine-Learning/blob/main/Python%20-%20Daya%20Analysis%20-%20Basics.ipynb)** 👇<br>

Python, *being the medium of conversation between us and the machine learning models*, demands solid foundation, as a pre-requisite,  in numerous programming concepts - such as 🔻
- Data structures (like `Lists`, `Dictionaries`, `Sets`, `Tuples`, etc.)
- Data types 
- Functions
- Logic / Control Flow
- Working with External Libraries
- 
- ... <i><b>and the list goes on !!!</b></i>
<br>

This notebook, in particular, should give you a good kickstart on clearing up your <i>Python</i> basics. Here, you will also be going to play with the functions (<i>methods</i>) of <b>[Numpy](https://numpy.org/doc/stable/index.html#)</b> - <i>"The fundamental package for scientific computing in Python"</i>, which shall give you the initial taste of <b>[Data Pre-processing](https://towardsdatascience.com/introduction-to-data-preprocessing-in-machine-learning-a9fa83a5dc9d)</b>. You will also experience the *"ease-of-formulating"* of complex mathematical functions using inbuilt <i>numpy</i> objects and methods. 
<br>

Next up, you will have **[Pandas](https://pandas.pydata.org/)** - <i>"A data analysis and manipulation tool,  built on top of Python programming language."</i> You will explore how to <i>"talk"</i> to a dataset. There are numerous inbuilt datasets in the Pandas library, but the  majority of the times, you will be working on external datasets - <u>downloaded</u> *OR* <u>extracted</u> from the web. Pandas also allows you for intermediate level data pre-processing like <i>String to Categorical datatype conversion, finding the min/max. value of an attribute, attribute drop / rename, etc.</i>
<br>

Possessing data and pre-preprocessing is just one of step of the many steps in data analysis. On later stage, you need to unravel <i>"What the data is conveying"</i>. And, for that, visual representation of the data is an added advantage. Here comes <b>[Matplotlib](https://matplotlib.org/)</b> - *"Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. It is widely used in data science, machine learning, and scientific computing to generate plots and graphs for data analysis and presentation."*  Whether it's a [heat map](https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html), [line plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html), [scatter plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html), [bar plot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html), matplotlib covers it all. We have another visualization library called <b>[Seaborn](https://seaborn.pydata.org/#)</b> - but here the question comes - ***"Why having 2 libraries for the same purpose ?"** - Seaborn complements Matplotlib by simplifying the creation of visually appealing statistical graphics with a more intuitive syntax and built-in themes. It provides specialized functions for complex plots like heatmaps and regression plots, integrates seamlessly with Pandas Dataframes, and automates many design aspects, reducing the need for extensive coding. This makes Seaborn ideal for quickly generating attractive and informative visualizations.* 

**[Scikit-Learn](https://scikit-learn.org/)** - *"is an open-source machine learning library for the Python programming language."* Whether it's a [classification](https://www.geeksforgeeks.org/getting-started-with-classification/) problem, dealing with a [regression](https://www.geeksforgeeks.org/regression-in-machine-learning/) dataset or need to put a [clustering](https://www.geeksforgeeks.org/clustering-in-machine-learning/) algorithm to work, scikit-learn contains a whole lot of varieties in algorithms to work with for different purpose and also provide data pre-processing functionalities (like splitting the data into *training* and *testing* components). In this notebook you will see the usage of **[Logistic Regression](https://www.geeksforgeeks.org/understanding-logistic-regression/)** algorithm  to predict the *"species"* of penguins using all the features in the **[Penguin](https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris)** dataset.

Lastly, you will have **[Optuna](https://optuna.org/)** - *"Optuna is an open-source software framework for hyperparameter optimization, written in Python. It is designed to be efficient, flexible, and easy to integrate with existing machine learning libraries and frameworks. Optuna aims to automate the process of hyperparameter tuning, which is crucial for improving the performance of machine learning models."* Here, Optuna is used to optimize hyperparameters like `C` (regularization strength),  `solver` (optimization algorithm) and `max_iter` (maximum number of iterations) for Logistic Regression models, aiming to improve performance metrics such as accuracy.
<br>
<br>

### **[Python - Data Analysis and Preprocessing](https://github.com/sricks404/Machine-Learning/blob/main/Python%20-%20Data%20Analysis%20%26%20Preprocessing.ipynb) 👇**<br>

Now that you are done grasping some knowledge about how to use some of the essential python libraries, it's time to hit the playground to play with datasets. 

> NOTE : The notebooks were run in `google colab environment`. If you want to try the same, just copy paste the below code in the starting cell of your notebook 🔻<br>

~~~
from google.colab import drive
drive.mount('/content/drive')
~~~
<br>

In this notebook, one of the most famous datasets has been used for the purpose of data analysis and preprocessing - <b>[penguins dataset](https://www.kaggle.com/code/parulpandey/penguin-dataset-the-new-iris)</b><br>

Data Preprocessing steps include 🔻<br>

- Handling Missing Entries<br>
- Handling Mismatched String Formats<br>
- Handling Outliers<br>
- Detecting and Managing Outliers within the dataset<br>
- Identifying Uncorrelated / Unrelated Features<br>
- Converting Features with String Datatypes to Categorical<br>
- Normalizing Non-Categorical Features<br>

> NOTE : It's always a good practice to export / save the preprocessed data because many a times, you shall be building your machine learning models in a separate notebook. As data preprocessing, if done on a huge dataset, consumes a significant amount of compute resources, thus ,a separate session shall be required for model building, training and evaluation.

<br>


### **[Data Analysis - Logistic Regression](https://github.com/sricks404/Machine-Learning/blob/main/Data%20Analysis%20-%20Logistic%20Regression.ipynb) 👇**<br>

Here, you will be encountering one of the first machine learning algorithm used in this repo - [logistic regression](https://www.geeksforgeeks.org/understanding-logistic-regression/). The dataset used is the preprocessed version of `penguins` dataset, whose original version was used for data analysis and preprocessing in the previous notebook.<br>

Below are the steps taken for training and evaluation of logistic regression model 🔻<br>

- Importing required libraries
- Loading the preprocessed dataset
- Choosing the target feature <b>(`y`)</b>
- Creating the data matrices for input data <b>(`X`)</b> and target data <b>(`y`)</b>
- Dividing the dataset into training and testing components
- Printing the shapes of the divided data components to confirm successful division
- Designing and implementing "logistic regression" model (according to the underlying mathematical formula and functions)
- Model Training
- Saving the weights of the model, which gave the  highest accuracy <i>(The evaluation metric can be changed according to your needs - especially te type of algorithm you chose and the type of problem you are solving)</i>
- Making prediction on <i>test dataset</i>
- Plotting the loss graph and printing out the loss values over each iteration<br>

> NOTE : When you try to save the weights of the model (either as a pickle `(.pkl)` file or as  HDF5 `(.h5)` file), make sure it can be successfully loaded again, say, for <i>model deployment</i>.

<br>


### **[Data Analysis - Linear Regression](https://github.com/sricks404/Machine-Learning/blob/main/Data%20Analysis%20-%20Linear%20Regression.ipynb) 👇**<br>

This notebook will tell you how to implement <b>[Linear Regression](https://www.ibm.com/topics/linear-regression#:~:text=Linear%20regression%20analysis%20is%20used,is%20called%20the%20independent%20variable.)</b> model from scratch, using the [Ordinary Least Square (OLS)](https://medium.com/@VitorCSampaio/understanding-ordinary-least-squares-ols-the-foundation-of-linear-regression-1d79bfc3ca35) method to <i>perform direct minimization of the squared loss function</i>. For this notebook, we will consider the preprocessed data obtained from the second part o fthe notebook - [Python - Data Analysis and Preprocessing](https://github.com/sricks404/Machine-Learning/blob/main/Python%20-%20Data%20Analysis%20%26%20Preprocessing.ipynb).<br>

Below are the steps followed 🔻

- Importing required libraries
- Loading the preprocessed dataset
		▶️ <i>If you think that the dataset is still not clean, you can again apply [data preprocessing techniques](https://github.com/sricks404/Machine-Learning/blob/main/Python%20-%20Data%20Analysis%20%26%20Preprocessing.ipynb) to make it useful.</i>
- 	Choosing the target feature <b>(`y`)</b>
- Creating the data matrices for input data <b>(`X`)</b> and target data <b>(`y`)</b>
- Dividing the dataset into training and testing components
- Printing the shapes of the divided data components to confirm successful division / splitting
- Calculating weights of the OLS equation
- Obtaining predictions and Calculating the sum of squared errors
- Plotting the `predictions Vs actual target feature values` graph
<br>

<i>In the later stage of this notebook, you will also observe the model implementation for <b>[Ridge Regression](https://www.ibm.com/topics/ridge-regression)</b>, including all the steps mentioned above.</i>
<br>

> NOTE : [scikit-learn](https://scikit-learn.org/stable/) or any other python library, that contain in-built models, were deliberately not used, just to give a clear idea how mathematically these algorithms can be / are implemented.

<br>

### **[Exploratory Data Analysis (EDA) - Regression](https://github.com/sricks404/Machine-Learning/blob/main/Exploratory%20Data%20Analysis%20(EDA)%20-%20Regression.ipynb) 👇**<br>

The content of this notebook constitute as one of the major component of the project - <u>***Suicide Rate Projection : A Holistic Approach with Mental Health Insights***</u>. This project was based on the prediction of suicide rate, accounting features like gender, age and mental health conditions, at a national level - by considering data from 100+ countries, encompassing numerous mental disorders. 

>NOTE : Even though the steps of *[EDA (Exploratory Data Analysis)](https://www.ibm.com/topics/exploratory-data-analysis)* are pretty general; it's up to you whether you want to divide a single step into multiple sub-steps or not.


Following were the techniques of EDA (Exploratory Data Analysis) incorporated🔻<br>

 - [Data Uniformity](https://towardsdatascience.com/data-uniformity-in-data-science-9bec114fbfae)
 - [Filling the Missing Values](https://www.analyticsvidhya.com/blog/2021/05/dealing-with-missing-values-in-python-a-complete-guide/)
 - Removing Differences in String Values - [Capitalizing String Values of a Feature](https://www.geeksforgeeks.org/string-capitalize-python/)
 - [Filling the Missing Values of the Columns that have Categorical Values (<i>using K-NN</i>)](https://machinelearningmastery.com/knn-imputation-for-missing-values-in-machine-learning/)
 -  [Removing unwanted Columns using Corelation Matrix (<i>Feature Selection</i>)](https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b)
 - [One-Hot Encoding](https://www.geeksforgeeks.org/ml-one-hot-encoding/)
 - [Setting Precision](https://www.geeksforgeeks.org/precision-handling-python/)
 - [Dataset Statistics](https://www.w3schools.com/python/pandas/ref_df_describe.asp#:~:text=The%20describe()%20method%20returns,The%20average%20(mean)%20value.)
 - [Detecting Outliers](https://www.geeksforgeeks.org/detect-and-remove-the-outliers-using-python/)
 - [Dataset Normalization](https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/)
 - Merging Columns with Proportionality Preservation
 -  Data Visualization🔻
	 - <i>"[Dist Plot](https://seaborn.pydata.org/generated/seaborn.distplot.html)</i>"
	 - <i>"[Scatter Plot](https://seaborn.pydata.org/generated/seaborn.scatterplot.html)"</i>
	 - <i>"[Histogram](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html)"</i>
	 - <i>"[Line Plot - Regression Plot](https://seaborn.pydata.org/generated/seaborn.regplot.html)"</i>
	 - <i>"[Violin Plot](https://seaborn.pydata.org/generated/seaborn.violinplot.html)"</i>
	 - <i>"[Joint Plot](https://seaborn.pydata.org/generated/seaborn.jointplot.html)"</i>
	 - <i>"[lmplot](https://seaborn.pydata.org/generated/seaborn.lmplot.html)"</i>
<br>

Below were the algorithms used for the [predictive analysis](https://www.ibm.com/topics/predictive-analytics)🔻

 - [Linear Regression](https://www.ibm.com/topics/linear-regression#:~:text=Linear%20regression%20analysis%20is%20used,is%20called%20the%20independent%20variable.)
 - [K-NN (K-Nearest Neighbour) Regression](https://www.ibm.com/topics/knn#:~:text=The%20k%2Dnearest%20neighbors%20(KNN,used%20in%20machine%20learning%20today.))
 - [ANN (Artificial Neural Network)](https://www.geeksforgeeks.org/artificial-neural-networks-and-its-applications/)
 - [Gradient Boosting (XGBoost - XGBRegressor)](https://machinelearningmastery.com/xgboost-for-regression/)
 - [Decision Tree Regression](https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html)
 - [Random Forest Regression](https://towardsdatascience.com/random-forest-regression-5f605132d19d)
<br>

As this was a "<i>Regression</i>" problem, the evaluation metrics used to measure the performance of the models were different from those, used in "<i>classification</i>". Here, the metrics used were🔻

 - [MSE (Mean Squared Error)](https://www.geeksforgeeks.org/mean-squared-error/)
 - [RMSE (Root Mean Squared Error)](https://c3.ai/glossary/data-science/root-mean-square-error-rmse/)
 - [R<sup>2</sup> - score (R-squared Score)](https://www.freecodecamp.org/news/what-is-r-squared-r2-value-meaning-and-definition/#:~:text=R%2DSquared%20values%20range%20from,50%25%2C%20and%20so%20on.)
 - [MAE (Mean Absolute Error)](https://medium.com/@m.waqar.ahmed/understanding-mean-absolute-error-mae-in-regression-a-practical-guide-26e80ebb97df)
<br>

### **[PyTorch - Neural Network (NN)](https://github.com/sricks404/Machine-Learning/blob/main/PyTorch%20-%20Neural%20Network%20(NN).ipynb) 👇**<br>
