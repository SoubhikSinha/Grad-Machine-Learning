
# Grad Machine Learning

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

This notebook has 2 parts - The first one goes with the explanation and implementation of a [Neural Network](https://www.ibm.com/topics/neural-networks) using [PyTorch](https://pytorch.org/) framework. The second part shows how to optimize the Neural Network on the basis of various hyperparameters. Below are the steps taken for the prediction🔻<br><br>
***Part 1 : Building a Basic Neural Network*** 🔽<br>
 - Loading the Dataset and examining Main Statistics
	 - Correlation Matrix (heatmap) - for numerical attributes / features
	 - Boxplot for the numerical attribute (`f3`)
	 - [Pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html) for feature (`f3`) with the target feature

 - Preprocessing and Splitting the Dataset
	- Handling missing values
	- Handling outliers using BoxPlots, [IQR (Inter Quartile Range)](https://statisticsbyjim.com/basics/interquartile-range/) and mean
	- Data Visualizations🔻
		 - Correlation Matrix (Heatmap)
		 - Histogram
		 - Pairplot

	- Converting categorical values to numerical values ([One-Hot Encoding](https://www.geeksforgeeks.org/ml-one-hot-encoding/))
	- Data Normalization using [StandardScaler (Scikit-learn)](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
	- Splitting dataset into train, validation and test components - [train_test_split(Scikit-Learn)](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
<br>

 - Defining the Neural Network<br>
	>  NOTE : PyTorch is deliberately used instead of TensorFlow, Keras, or Scikit-learn because it allows you to understand the underlying structure of how a neural network is formed. This provides significant flexibility, experimentation opportunities, debugging capabilities, and performance benefits.

<br>

 - Training, Validation and Testing of the Neural Network
 
	 - Setting the no. of [epochs](https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-epoch-in-machine-learning) and [batch size](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
	 - Setting the criterion ([loss function](https://www.datacamp.com/tutorial/loss-function-in-machine-learning)), optimizer and learning rate
	 - Creating Dataloaders
	 - Storing the training and validation - *accuracy* and *loss* values for every epoch
	 -  Testing the model and recording the score of evaluation metrics🔻
 
		 - Accuracy
		 - Precision
		 - F1 score
		 - Recall (Sensitivity)
<br>

 - [Saving the weights](https://pytorch.org/tutorials/beginner/saving_loading_models.html) of the trained, validated and tested model<br>
 - Visualizing the results🔻
	 - [Confusion matrices](https://www.analyticsvidhya.com/blog/2020/04/confusion-matrix-machine-learning/#:~:text=A%20Confusion%20matrix%20is%20an,by%20the%20machine%20learning%20model.) (heatmap)
	 - [ROC Curve](https://medium.com/@shaileydash/understanding-the-roc-and-auc-intuitively-31ca96445c02)
	 - Comparative visualization on training, validation and testing accuracy scores
	 - Comparative visualization on training, validation and testing loss values
 <br><br>

***Part 2 : Optimizing the Neural Network*** 🔽<br>

 - Choosing the hyperparameters to optimize the Neural Network🔻
	- [Dropout rate](https://spotintelligence.com/2023/08/15/dropout-in-neural-network/#:~:text=The%20dropout%20rate%20typically%20ranges,to%20the%20computation%20of%20activations.)
	- [Optimizer](https://ml-cheatsheet.readthedocs.io/en/latest/optimizers.html#:~:text=They%20tie%20together%20the%20loss,by%20futzing%20with%20the%20weights.)
	- [Activation Function](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
	- [Initializer](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/)
<br>

 - Various other methods employed for optimization🔻
 	 - [Early Stopping](https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/)
 	 - [K-Fold](https://machinelearningmastery.com/k-fold-cross-validation/)
 	 - [Learning Rate Scheduler](https://machinelearningmastery.com/using-learning-rate-schedule-in-pytorch-training/)
 	 - Introduction of  [Batch Normalization](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739)
 	<br>
 - Obtaining the visualization results
 
*`Via the abovementioned steps, one can easily compare and conclude by declaring which model setup (base model or one among the optimized models) is superior / dominant`.*

<br>

### **[PyTorch - Convolutional Neural Network (CNN)](https://github.com/sricks404/Grad-Machine-Learning/blob/main/PyTorch%20-%20Convolutional%20Neural%20Network%20(CNN).ipynb) 👇**<br>

This is in continuation to the [previous notebook](https://github.com/sricks404/Grad-Machine-Learning/blob/main/PyTorch%20-%20Neural%20Network%20(NN).ipynb) - thus, should be considered as Part 3. In this notebook, you will be observing how to build a [Convolutional Neural Network (CNN)](https://www.geeksforgeeks.org/introduction-convolution-neural-network/) model using the PyTorch framework. The dataset considered here is a subset taken from [MNIST dataset](http://yann.lecun.com/exdb/mnist/). In Part 4, a CNN architecture, [VGGNet - 11](https://medium.com/@siddheshb008/vgg-net-architecture-explained-71179310050f), is implemented. Below are the steps taken for model implementation🔻<br><br>

***Part 3 : Building a CNN*** 🔽<br>
- Loading, preprocessing, analyzing, visualizing and preparing the dataset for training
	 - One Hot Encoding
	 - [Image Normalization](https://medium.com/@shoaibrashid/what-is-image-normalization-d8305bf328c0)
	 - Obtaining `Main Statistics`
	 - Data Visualization🔻
		 - Data Distribution among classes
		 - Histogram for Pixel Intensities
	
	 - Creating Training, Validation and Testing components - using `train_test_split()`
 
- Building and Training basic CNN architecture<br>
   > NOTE : Here, we have considered a maximum of 10 hidden layers, just to keep the model complexity up to a limit. Also, you can observe how [CUDA - GPU](https://developer.nvidia.com/cuda-gpus) is used as a hardware accelerator to expediate the CNN's training process. Below is a code snippet to give you an idea on how to use GPU in Machine Learning models🔻

    ~~~
	# Checking if a GPU (cuda) is available, and setting the device accordingly
	device  =  "cuda" if torch.cuda.is_available() else "cpu"

	# Printing the selected device
	print("Selected device:", device)

	# Defining the CNN architecture
	class SelfCNN(nn.module):
		.
		.
		.
		
	# Initializing the model
	num_classes = 36
	model = SelfCNN(num_classes).to(device)

	# Printing the model summary
	print(model)
    ~~~

 - Model Training
	- Creation of Data Loaders
	- Setting criterion and optimizer
	- Model Tuning methods used🔻
		- Early Stopping
		- Learning Rate Scheduler
	- Recording Training and Validation - Accuracy and Loss Scores for each epoch

- Model Testing
	- Evaluation Metrics Used🔻
		- Accuracy
		- Precision
		- F1-score
		- Recall (Sensitivity)

- Saving the weights of the trained CNN model using PyTorch

 - Visualizing the results🔻
	 - Confusion matrices (heatmap)
	 - ROC Curve
	 - Comparative visualization on training, validation and testing accuracy scores
	 - Comparative visualization on training, validation and testing loss values

<br>

***Part 4 : VGG-11 Implementation*** 🔽<br>
For reference, you can refer to the research paper - [VGG Architecture](https://arxiv.org/abs/1409.1556) to observe how the VGGNet-11 (Version A) is implemented with respect to the given architecture in the paper. The model implementation, training-validation-testing stage and visualization result's code remains almost the same as CNN.<br><br>

### **[CIFAR10 - Image Classification (CNN)](https://github.com/sricks404/Grad-Machine-Learning/blob/main/CIFAR10%20-%20Image%20Classifier%20(CNN).ipynb) 👇**<br>

The **[PyTorch](https://pytorch.org/)** documentation is a fantastic resource for gaining practical knowledge about deep learning techniques. On the PyTorch website, they offer a course called : **[Deep Learning with PyTorch : A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)**, which will help you get started quickly on how to use PyTorch. which is designed to help you get started quickly with using PyTorch. The goal of this course/tutorial is to "***Train a small neural network to classify images***". There are 4 0notebooks you need to go through and run each cell of everynotebook to understand how everything works in PyTorch. I have only mentioned the last notebook - ***Training a Classifier***, to give you an idea how an **[Image Classification](https://huggingface.co/tasks/image-classification#:~:text=Image%20classification%20is%20the%20task,class%20the%20image%20belongs%20to.)** problem can be solved using CNN on a large image dataset like **[CIFAR 10](https://www.cs.toronto.edu/~kriz/cifar.html)**. Below are the steps followed in this notebook 🔽

 - Load and Normalize CIFAR 10
 - Define a Convolutional Neural Network
 - Define a Loss Function and Optimizer
 - Train the Network
 - Test the Network on the Test Data
 - ***Training on GPU***
 - Hyperparameter Changing
 - [Optuna](https://optuna.org/) Tuning
<br>

### **[Reinforcement Learning - SARSA - Double-Q SARSA - n-step SARSA](https://github.com/sricks404/Grad-Machine-Learning/blob/main/Reinforcement%20Learning%20-%20SARSA%20-%20Double-Q%20SARSA%20-%20n-step%20SARSA.ipynb) 👇**<br>


**[Reinforcement Learning](https://www.geeksforgeeks.org/what-is-reinforcement-learning/#)** is a way for computers to learn how to make decisions, similar to how humans learn through experience. Imagine teaching a dog new tricks. You give it treats when it does something right and withhold treats when it doesn't. Over time, the dog learns to perform the tricks that earn it treats. Let us take an example 🔽
<br><br>
In the context of reinforcement learning, the "dog" is the agent, and the "tricks" are actions it can take. The "treats" are rewards it gets from the environment (which could be anything from a simple game to a complex real-world scenario) when it makes the right choices. The agent tries different actions, sees what works, and gradually learns to choose actions that maximize its rewards.
<br>
The Key elements in Reinforcement Learning (talking with respect to the above mentioned example) are as follows🔻

 - **Agent** : The learner, like our dog, trying to figure out what to do.
 - **Environment** : The world in which the agent operates, like the living room where the dog learns tricks.
 - **State** : The current situation or condition in the environment, like the dog being in front of you.
 - **Action** : The choices the agent can make, such as sitting, standing, or rolling over.
 - **Reward** : The feedback the agent gets after taking an action, like getting a treat or a pat on the head.

Through this process, the agent aims to maximize the total rewards it gets over time, learning the best actions to take in different situations. This learning approach is widely used in various applications, from game playing to robotics and even financial trading.
<br><br>
Now let us talk about the algorithms that you will find in this notebook 🔽
<br><br>

**[SARSA (State-Action-Reward-State-Action)](https://www.geeksforgeeks.org/sarsa-reinforcement-learning/)** 🔻
<br>

This algorithm is a is a way for a computer program (an agent) to learn how to make better decisions based on its experiences. Think of it like learning to play a new board game. Here's how it works in a more human-friendly way ⏬
 - **State ( S )** : This is like the current position of the game pieces on the board.
 - **Action ( A )** : The player (agent) decides what move to make next, like choosing to roll the dice or move a piece.
 - **Reward ( R )** : After making the move, the player sees the result, such as moving closer to the goal or getting a penalty.
 - **Next State ( S' )** : The game board changes based on the move, leading to a new situation.
 - **Next Action ( A' )** : The player then thinks about what to do next in this new situation.

In SARSA, the agent keeps track of these steps to learn what moves work best. It remembers what it did, what happened next, and how good or bad the outcome was. Over time, the agent uses this information to get better at the game.
<br>
<br>
So, SARSA helps the agent learn from its own experiences, constantly improving its strategy by considering the actions it actually takes and the outcomes that follow. This way, it becomes better at choosing moves that lead to more rewards, much like how we learn to play games better by practicing and learning from our mistakes.
<br>
<br>
<br>

**[Double Q-SARSA](https://towardsdatascience.com/double-deep-q-networks-905dd8325412)** 🔻
<br>

This is a variant of Q-Learning that aims to reduce the overestimation bias that can occur in Q-Learning. Double Q-Learning achieves this by using two separate estimators (or Q-tables) for the action-value function. The idea is to decouple the action selection from the action evaluation to provide more accurate value estimates.
<br>
Double Q-Learning uses two separate Q-tables, *Q<sub>1</sub>* and *Q<sub>2</sub>* ​, to store the Q-values. During each update, it randomly chooses one of the Q-tables to update using the action value from the other table to decouple the action selection from the action evaluation. The update rule for Double Q-Learning is :

 - With probability 0.5, update *Q<sub>1</sub>(S<sub>t</sub> , A<sub>t</sub>)* using  Q<sub>2</sub>​ to estimate the maximum Q-value :
 
	 <i>Q<sub>1</sub>​(S<sub>t</sub>​, A<sub>t</sub>​) ← Q<sub>1</sub>​(S<sub>t​</sub>, A<sub>t</sub>​) + α[R<sub>t+1</sub> ​ +γQ<sub>2​</sub>(S<sub>t+1</sub>​, arg max<sub>a</sub> ​Q<sub>1</sub>​(S<sub>t+1</sub>​, a)) − Q<sub>1</sub>​(S<sub>t</sub>​, A<sub>t</sub>​)]</i>
<br>

 - With probability 0.5, update Q<sub>2</sub>(S<sub>t</sub>, A<sub>t</sub>) Q<sub>2</sub>​(S<sub>t</sub>​, A<sub>t</sub>​) using Q<sub>1</sub>​ to estimate the maximum Q-value :


	<i>Q<sub>2</sub>​(S<sub>t</sub>​, A<sub>t</sub>​) ← Q<sub>2</sub>​(S<sub>t</sub>​, A<sub>t</sub>​) + α[R<sub>t+1</sub> ​+γQ<sub>1</sub>​(S<sub>t+1</sub>​, arg max<sub>a</sub>​Q<sub>2</sub>​(S<sub>t+1</sub>​, a)) − Q<sub>2</sub>​(S<sub>t​</sub>, A<sub>t</sub>​)]</i>
	
<br>

In Double Q SARSA, a similar approach can be applied to reduce overestimation in SARSA, but instead of the maximum Q-value estimation, it would involve the Q-value updates based on the chosen actions.
<br>

The general idea is to have two action-value functions, Q<sub>1</sub>​ and <sub>Q2</sub>​, and use them alternately to update each other's estimates. This helps mitigate the problem of overestimating action values, leading to more accurate learning.
<br>

In summary, Double Q-SARSA uses two separate Q-tables to reduce overestimation bias, similar to Double Q-Learning, but within the SARSA framework where the next action is also considered in the update process.
<br>
<br>
<br>

**[n-step SARSA](https://medium.com/zero-equals-false/n-step-td-method-157d3875b9cb)** 🔻
<br>

n-step SARSA is a reinforcement learning algorithm that extends the SARSA method by considering multiple steps (n steps) of actions and rewards before updating the Q-value of a state-action pair. Instead of updating based on a single action and its immediate reward, n-step SARSA looks at a sequence of n actions and the cumulative rewards from those actions.
<br>

In simple terms, n-step SARSA provides a more comprehensive view of the future by considering more steps ahead, which can help the agent learn more stable and accurate value estimates. This method balances short-term and long-term planning in the agent's decision-making process.
<br>
<br>

Now that we have an overview of the algorithms, let us dive into the notebook 🔽<br><br>

**Part I : Define an RL Environment** 🔻
<br>

Here, you will see how an RL environment has been setup for the game of *"Treasure Hunt"*. You can check the first code cell of the notebook to acquire information about the actions and rewards set for the agent and environment. The goal : *Finding the treasure with the largest cumulative reward while dodging / passing over obstacles is the agent's mission.*
<br>
<br>

**Part II : Implement SARSA** 🔻
<br>

Here, the implementation of SARSA Algorithm will be applied for the environment set in Part I.
<br>
<br>

**Part III : Implement Double Q-Learning** 🔻
<br>

Here, the implementation of Double Q-Learning Algorithm will be applied for the environment set in Part I.
<br>
<br>

**n-step Bootstrapping (n-step SARSA)** 🔻
<br>

In reinforcement learning, this updates value estimates using the returns from multiple future steps (n steps) instead of just one. It combines observed rewards from these steps with an estimated future value to make more informed updates. This method balances short-term and long-term planning, providing a mix of immediate feedback and future predictions for more accurate learning.
<br>
<br>

***n-step SARSA*** is like learning from a series of moves in a game instead of just one move at a time. Imagine you're playing a board game and want to figure out the best strategy. Instead of just looking at the immediate result of your next move, you consider what happens over the next few moves.
<br>

Here’s how it works :

 - **Play Multiple Moves** : The agent makes a sequence of moves, seeing the rewards and changes in the game over those steps.
 - **Add Up the Results** : It then adds up all the rewards from these moves to get a big-picture view of how good the strategy was.
 - **Update Strategy** : Finally, it updates its strategy based on this larger view, improving how it plays in similar situations in the future.
<br>

By looking at more than just the immediate next move, n-step SARSA helps the agent learn more effectively from its experiences.
