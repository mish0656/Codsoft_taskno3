# Codsoft_taskno3
Iris Flower Classification
The dataset is already provived by the codsoft from kaggle.
Explanation of this task is given below:

1. Importing Libraries:
   - `pandas`: Used for data manipulation and analysis.
   - `matplotlib.pyplot`: Used for creating visualizations like scatter plots.
   - `seaborn`: A library based on Matplotlib for creating attractive statistical graphics.
   - `warnings`: Used to manage warning messages.
   - `train_test_split` from `sklearn.model_selection`: Used to split the dataset into training and testing sets.
   - `LogisticRegression` from `sklearn.linear_model`: Used to create a logistic regression model.
   - Various functions from `sklearn.metrics`: Used to evaluate model performance.

2. Load and Explore Data:
   - The code loads the Iris dataset from a CSV file using `pd.read_csv`.
   - It then displays the first few rows of the dataset using `iris_d.head()`.
   - The summary statistics of the dataset are displayed using `iris_d.describe()`.
   - The count of unique species in the dataset is shown using `iris_d['species'].value_counts()`.
   - Null values in the dataset are checked using `iris_d.isnull().sum()`.

3. Scatterplot Visualization:
   - A scatter plot is created to visualize the relationship between 'petal_length' and 'sepal_width' for each species.
   - Three different colors ('orange', 'blue', and 'pink') are used to represent three species ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica').
   - The scatter plot is labeled and displayed using `plt.scatter`, `plt.xlabel`, `plt.ylabel`, and `plt.legend`.

4. Data Preprocessing:
   - The `LabelEncoder` from `sklearn.preprocessing` is used to encode the target variable 'species' into numeric labels.
   - The encoded labels replace the original species values in the dataset using `le.fit_transform`.

5. Train-Test Split:
   - The features (sepal_length, sepal_width, petal_length, and petal_width) are extracted as input 'x'.
   - The target variable (species) is extracted as output 'y'.
   - The dataset is split into training and testing sets using `train_test_split`.

6. Model Training:
   - A logistic regression model is initialized using `LogisticRegression`.
   - The model is trained on the training data using `model.fit`.

7. Model Evaluation:
   - The accuracy of the model on the training data is calculated using `model.score`.
   - The model's predictions are made on the training data using `model.predict`.
   - The classification report and confusion matrix are generated using `metrics.classification_report` and `metrics.confusion_matrix`, respectively.

8. Making Predictions:
   - Two test data points are provided (`d_points` and `d_pts`).
   - The model predicts the species of each data point using `model.predict`.
   - The predictions are printed in the output.
  
     I have done all the related things and model is predicting accurately . 

