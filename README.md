## Data Mining of Legacy Data

<h3> Description of the task. </h3>

- Introduction: Practical experience in handling a data-mining project using industry tooling for legacy data. The Data provided as part of this assessment is from the domain of cardio vascular medicine. The work herein requires the description and analysis of data for the given domain, including manipulation of data in various forms, as well as creation of classifiers

- Software: Python and Jupyter

- Methodology: CRISP-DM 

<h2>Methodology</h2>

<h3>Business Understanding</h3>
The first step to be undertaken in the project was business understanding of the data. The sample given was medical data with 11 columns. The first 10 columns would be used to predict the label value using a machine learning model. Looking at the data types of the data, Random and IPSI was a float64, Id was a int and the rest were objects. A model would be chosen and created to take a sample of this data and then predict the label of the test data.

<h3>Data Understanding</h3>
Doing a cat plot of label with a hue of each data column, the data was visualized. The data showed that patients with diabetes were more likely to be at risk than those without. Those with diabetes and not at risk was very small. There was also a bar for unknown suggesting that the data had to be cleaned.
A visualization of IPSI using cat plot showed that those at risk generally had higher IPSI but were also distributed sparsely along lower values. Those not at risk had the highest count near the median.
Indication showed that it also had null or unknown values. ASX is split into two columns because each spelling of the entry is treated as a unique entry. This would have to be fixed in data preparation. The entries that had TIA were more likely to not be at risk than they were to be at risk.
In IHD those with IHD seem more likely to no be at risk. Whilst those that do have it have an almost even chance at being at risk or not being at risk. The same also seems to apply to Hypertension.
Patients that have Arrhythmia seem very likely to be at risk. History however seems to not matter that much.
BY doing a box plot of IPSI it is shown that IPSI has a small interquartile range between 73 and 85. The mean IPSI value is 78.8 with a standard deviation of 10.16.There lay a few outliers below the minimum.
Inspecting the data as a data frame and looking at the top 30 values and then identifying the number of unique values, Random appears to repeat as a value whereas ID does not. In the specification Session is mentioned as a variable however it is not present in the data whereas Random is not properly explained. A reasonable explanation for this would be that Random which has repeat values is the ID of the patient, with this value repeating as a patient has a second or third session. By inspecting the data it seems the same Random value generally repeats once and infrequently twice. The Id value however does not repeat which may be because this is the session identifier and would always be unique.

<h3>Data Cleaning</h3>
The data is checked for null values and quite a few appear. These are inspected individually for each category. Diabetes has 2 null values, Indication has 3, Hypertension has 3, History has 2, IPSI has 4, label has 4. Each set of null values has no correlation between them thus an explanation for why they are null such as certain entries being less likely to be entered are not applicable in this case. The values seem to be Missing Completely at Random (MCAR). MCAR means that a certain value being missing has nothing to do with its hypothetical value and with the values of other variables. Apart from IPSI the values missing are yes or no values. Imputing these values would rely on guesswork as there are no averages apart from mode to go on. Imputing the most common value into these spaces
600092 Data Mining and Decision Systems
would create a biased training data set and possibly incorrect results when using train test split so it is better to drop the rows of values as the dataset is quite large. Entries with IPSI missing are also being dropped to not introduce biases into the model as there is no easy way to predict the most accurate value of the missing values. These entries that have missing data likely occurred due to human error rather than the individual patient choosing to not disclose this information as the information was likely collected by a professional rather than done using a survey.
Contra has broken when trying to use it as a number value to display graphs, after inspecting it, it appears to be a object rather than a float or int so it will have to be converted to a float.
All the null values are dropped from the dataset correctly. After identifying a space in Contra for one of the entries it is replaced with a NaN and then also dropped.
Looking at the unique values of label there is a third value called Unknown. There is no description of what this value should be or any correlation between the entries that contain it so the entries containing this value are dropped. Contra values in the data frame are converted into float values.

<h3>Data Preparation</h3>
Now that the Contra value is fixed it can be visualized properly. Testing the data in a scatter plot against IPSI shows that it is working and also shows that those that have high values in both are labeled with risk. Having a low contra and high IPSI does not indicate that the patient is at risk with this data set. Contra has a very high interquartile range so it cannot be guessed what value would be the most appropriate to assign it the value 1 when fed in to a multi-layer perceptron. IPSI also cannot be easily assigned a 0 or 1 value.
To prepare the data for the multi-layer perceptron Random and Id are dropped. Even if they were renamed to ID and Session respectively due to the very low amount of repeat values in ID it is not apparent how they would be beneficial for the model being tested. Thus both values as well IPSI and Contra are being dropped and not used in this model. Label is dropped to add it to the end of the data frame to use as the y data. The data that the model is trying to predict. Indication was also dropped because it has to be split into four values using the get dummy method.
After dropping the relevant columns the data has the columns Diabetes, IHD, Hypertension, Arrhythmia and History and all the data in the data frame has the datatype object.
Using the magic command time it each value that is no in the data frame is replace with a 0 and using an else statement the remaining values become 1. As there are only yes or no values on the data frame this works well.
Indication is split using get dummy and is then merged to the data frame along with label.


### Model Selection

<h3>Multi-Layer Perceptron<h3>
Multilayer perceptron refers to a neural network with at least three layers of nodes, an input layer, some number of intermediate layers, and an output layer. Each node in a given layer is connected to every node in the adjacent layers.
A perceptron is meant to simulate a neuron in that it receives inputs like signals that are received by dendrites. Inputs are then processed, this is the cell processing the signals. A output is then created. The output is then received by the next neuron like a dendrite. These neurons are set up as a network. The input layer, hidden layer and the output layer. The hidden layer acts as a distillation layer that distills the patterns that are important from the inputs and sends it to the next layer.

<h3>Modelling</h3>
The data has to be split between what is being tested and what is being predicted. Thus the X value is assigned to the data of the first 10 columns and the last column, label, is assigned to y.
The original dataset should be split up into training and testing data. 70% of the data is to used for training and the remaining 30% will be used for testing the models ability to predict the label of the data. The data is split so that there is data for the model to be evaluated on to see how well the model performs on unseen data. Test data has to remain unseen by the model. A scaler is initialized to scale the data so it can easily be fed into the model. The data to be trained and the data to be tested is then scaled. MLP is imported and a class is initialized to set it up. Using fit the model is trained on the x train and y train values. Using predict the model uses the training data to predict the y values of the test data. Using metrics from sklearn classification report and confusion matrix is imported to get an understanding of the performance of the model. A confusion matrix also known as error matrix is a predictor of model performance on a classification problem.
After importing them a classification report is created and a confusion matrix. This outputs the results of the model as a confusion matrix and gives the precision, recall, f1-score and support of the values predicted. The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative. The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples. The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The accuracy score of this initial model fluctuates around 0.88.
Evaluation
When preparing the data for the model the two columns for ASX were not dealt with properly as this is a clear problem which may affect the performance. Testing different iterations for the value showed that 1000 iterations is the optimum amount as any higher or lower reduce the accuracy.
Data Understanding
Asx and Asx have been entered as separate and unique values and this was likely a human error so the two columns combining would have a positive effect on the data.
Data Preparation
After creating a dummy of indication another column is created called ASX that add the two ASX variables together and then merges with the rest of the data frame after removing the redundant columns. The column is tested using value counts to not have lost any values in the transition.
Modelling
A new multi-layer perceptron is created after X value uses the combined values of ASX now. Running the model on the data again the model outputs an accuracy result of around 0.92. The model is teste for the best number of layers.
Evaluation
The accuracy of the data has improved slightly. The best layers are 10, 10,10 for the model. The model needs to be tested for the optimum test size.

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/haych-hub/Mining-of-Legacy-Data/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
