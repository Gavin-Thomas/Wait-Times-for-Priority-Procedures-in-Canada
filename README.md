
# <p align="center">Canadian Healthcare Analysis: Wait Times for Priority Procedures</p>
### <p align="center">By Gavin Thomas</p>
_____

<p align="left">Dataset: Canadian Institute for Health Information (CIHI). Wait Times for Priority Procedures in Canada — Data Tables. Ottawa, ON: CIHI; 2022.</p>

Dataset [HERE](https://www.cihi.ca/sites/default/files/document/wait-times-priority-procedures-in-canada-data-tables-en.xlsx)

SKILLS I DEMONSTRATE:
- Data cleaning with Pandas
- Data visualization with Seaborn and Matplotlib
- Machine learning with Scikit-Learn
- Machine learning with PyCaret
_____

## Intro

Contained in this repository is my data analysis of the 2023 CIHI wait times data table for priority procedures. This CIHI dataset does not contain ALL types of priority procedures; rather, the data table contains a few different types of procedures that may give a general idea of waiting times depending on the procedure type, and the province of residence. 

The reason I chose this dataset is two-fold:
1. The Canadian health-care system is known to have long wait times.
2. The CIHI has released a clear, well-formatted dataset.

When I did my data analysis I did not include all parameters. The reason I did this was to simplify my life, and to reduce any confounders to the random forest regressor I built, as well as to generate clear graphs.

In this readme file, I will take you through my process of data cleaning, analysis/visualization, and model building. I have also attached my code for data cleaning, the cleaned csv file, my code for data visualization (some graphs added later on- so not fully up to date), and my code for the random forest regressor.

---
## Data Cleaning

Firstly, I converted the [CIHI dataset](https://www.cihi.ca/sites/default/files/document/wait-times-priority-procedures-in-canada-data-tables-en.xlsx) to a CSV file. I only added the last sheet of the Excel dataset to my CSV file.
GOAL METRIC: Column "Indicator Result" (called "Result" in cleaned CSV)

Everything I Cleaned:
- Deleted columns "Region", "Reporting Level", and "Unit of Measurement" (I did this one later on)
- Deleted rows containing "Canada"
- Only rows I did not delete from the Metric column was the "50th percentile" (50th percentile wait time)
- Changed all Provinces to their abbreviation (ex. AB instead of Alberta)
- Deleted all rows containing "Proportion" under column  "Unit of Measurement"
- Deleted all Data years past 2021 (not including 2022)
- Deleted all Data years with letters (ie. 2019FY)
- Changed All "Indicator Results" Column "Hours" to "Days" by dividing by 24.
- Changed Column Title "Province/Territory" to Province
- Changed Column Title "Data year" to "Year
- Changed Column Title "Indicator Result" to "Result"

Cleaned CSV file [HERE](https://github.com/Gavin-Thomas/Wait-Times-for-Priority-Procedures-in-Canada/blob/main/cleaned-patient_wait_times.csv)

I manually deleted the columns that I didn't want. These were "Region" & "Reporting Level". Later on I deleted "Unit of Measurement"

I used pandas and numpy for data cleaning.

```
import pandas as pd
import numpy as np

# I then read the CSV using pandas, and got a header of the file

df = pd.read_csv("unclean_data.csv")
df.head(5)
```
The file did not look pretty at this point, so I dropped all N/A values.
```
df = df.dropna()
```
Next, I set a new variable for each item I wanted to remove. Each variable built off the subsequent variable. I could have made this a much smaller block of code, but for readability, it is layed out like so. Then, using the "~", I took the inverse and only saved items that DID NOT contain what I specified. I used the lambda command to find the specific usage of the word, and then eliminated it with the str.contains function. Axis=1 was used to represent the row that I deleted. 

Then I saved my changes from the last variable to a new file called unclean_data2.csv

I also did the EXACT same thing with rows containing "Proportion" and "Number of Cases" and "Volume" - not shown.
```
# Dropping any rows listed Canada, Q, Proportion, Number of Cases, FY

new_df2 = df[~df.apply(lambda row: row.astype(str).str.contains("Canada").any(),axis=1)]
new_df3 = new_df2[~new_df2.apply(lambda row: row.astype(str).str.contains("Q").any(),axis=1)]
new_df4 = new_df3[~new_df2.apply(lambda row: row.astype(str).str.contains("Proportion").any(),axis=1)]
new_df5 = new_df4[~new_df2.apply(lambda row: row.astype(str).str.contains("Number of cases").any(),axis=1)]
new_df6 = new_df5[~new_df2.apply(lambda row: row.astype(str).str.contains("FY").any(),axis=1)]

# Then I saved to a new file

new_df6.to_csv("unclean_data2.csv")

# Note: I also did this with "90th percentile", and "Volume" (not shown).
```
Next, I reloaded the new CSV file and in any row that contained the word "Hours", I modified the neighboring value under the column "Indicator Result" to show hours instead of days by dividing by 24. I used the ".loc" function to identify the position of hours, and then used "/=" to divide the number of hours by 24 to get days.

```
# Changing hours to Days using loc function and index function

df.loc[data.index.str.contains('Hours'), 'Indicator Result'] /= 24

# As an example, for all the provinces. (Other provinces not shown)
# I changed the occurence of the word Alberta with AB using the loc function

data.loc[:, 'provinces'] = data['provinces'].replace('Alberta', 'AB')
```
Then finally, I deleted rows later than 2019, and renamed columns as "Result" (TARGET VARIABLE - which is wait time in days), Indicator, Province, and Year. 

Once done all this, I saved the file to the final copy of the cleaned CSV titled: **patient_wait_times.csv**

---
## Data Visualization 

### Provincial Wait Times
```
df = pd.read_csv("patient_wait_times.csv")

avg_wait_time = df.groupby('Province')['Result'].mean()

# Print wait time based on Province
print(avg_wait_time)

# RESULT (WAIT TIME IN DAYS)

Province
AB      74.457611
BC      83.642558
MB     105.602222
NB      79.919872
NL      65.843298
NS     110.810046
ON      70.604351
PEI     69.459600
SK      81.135310
```
It appears that on average, NL has the shortest wait times and Nova Scotia has the longest wait times. Let's visualize the data to get a better understanding.

For this next part, I use Seaborn.
```
sb.set(rc={'figure.figsize':(25,10)})

violin_plot = sb.violinplot(y="Result",x="Province",data=df)
violin_plot.set_xlabel('Indicator', fontsize=40, fontfamily = 'Times New Roman')
violin_plot.set_ylabel('Result (Days)', fontsize=40, fontfamily = 'Times New Roman')
violin_plot.tick_params(axis='both', which='major', labelsize=25, width=2, length=6, pad=8)
```
![ViolinPlot-Province](https://github.com/Gavin-Thomas/Wait-Times-for-Priority-Procedures-in-Canada/blob/main/Images/ViolinPlot-Province.png?raw=true)
_Figure 1_. Violin plot of wait time in days for each Canadian province (excluding territories) for priority procedures. 

Next up is the barplot...
```
barplot = sb.barplot(data=df, x="Province", y="Result")
barplot.set_xlabel('Province', fontsize=50, fontfamily = 'Times New Roman')
barplot.set_ylabel('Result (Days)', fontsize=50, fontfamily = 'Times New Roman')
barplot.tick_params(axis='both', which='major', labelsize=25, width=2, length=6, pad=8)
```

![BarPlot-Provice](https://github.com/Gavin-Thomas/Wait-Times-for-Priority-Procedures-in-Canada/blob/main/Images/BarGraph.png?raw=true)
_Figure 2_. Bar plot of wait time in days for each Canadian province (excluding territories) for priority procedures. SEM bars included.

### Wait Times Based on Year
Note:
- Yearly wait time trends based off data from 2008 - 2021

```
import matplotlib.pyplot as plt
import seaborn as sb
# Set figure size
sb.set(rc={'figure.figsize':(40, 16)})

# Set Barplot
barplot = sb.barplot(data=df, x="Year", y="Result")
# Set X label 
barplot.set_xlabel('Year', fontsize=50, fontfamily = 'Times New Roman')
# Set Y Label
barplot.set_ylabel('Wait Time (Days)', fontsize=50, fontfamily = 'Times New Roman')
# Set label sizes
barplot.tick_params(axis='both', which='major', labelsize=25, width=2, length=6, pad=8)
```

![Year-Wait-Time](https://github.com/Gavin-Thomas/Wait-Times-for-Priority-Procedures-in-Canada/blob/main/Images/Wait_Time_Year.png?raw=true)

_Figure 3_. Barplot of yearly wait time trends for priority procedures in Canada from the year 2008 to 2021.

### Wait Times Based on Indicator (Procedure)


```
# Take the wait time based off indicator (procedure) and result (days)
avg_wait_time2 = df.groupby('Indicator')['Result'].mean().round(1)

print(avg_wait_time2)

# MEAN WAIT TIME (DAYS) BASED OFF PROCEDURE TYPE
Indicator
Bladder Cancer Surgery                          24.6
Breast Cancer Surgery                           17.8
CABG                                             8.5
CT Scan                                         17.8
Cataract Surgery                                84.4
Colorectal Cancer Surgery                       19.2
Hip Fracture Repair                              1.0
Hip Fracture Repair/Emergency and Inpatient      1.1
Hip Replacement                                126.6
Knee Replacement                               157.7
Lung Cancer Surgery                             24.0
MRI Scan                                        53.6
Prostate Cancer Surgery                         40.9
Radiation Therapy                                9.5
Name: Result, dtype: float64
```
![Wait-Time-Indicator](https://github.com/Gavin-Thomas/Wait-Times-for-Priority-Procedures-in-Canada/blob/main/Images/Wait_Time_Indicator.png)
_Figure 4_. Wait time (days) based off the type of procedure (called "indicator").

---
## Random Forest Regressor

To cap off this project, I wanted to built a model to predict somemone's wait time for a priority procedure depending on their province, indicator (ie. disease/procedure needed), and data year.

I used a random forest regressor, since it is a fairly simple model to build relative to other ML models.

```
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# First, I load the data
df = pd.read_csv("cleaned-patient_wait_times.csv")

# Next, I converted the "Days" column to numeric so I didn't get any errors
df['Result'] = pd.to_numeric(df['Result'], errors='coerce')

# Then I encoded categorical variables using dummy variables
df = pd.get_dummies(df, columns=["Province", "Indicator", "Metric", "Year"])

# Then I Split the data into training and testing sets.
# I used a 80/20 split (80% training, 20% testing)

X = df.drop(["Result"], axis=1) # drop the target variable
y = df["Result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest regressor and fit it to the training data
rf = RandomForestRegressor(n_estimators=91, random_state=38)
rf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf.predict(X_test)

# Reshape y_pred to match the dimensions of y_test
y_pred = np.reshape(y_pred, (len(y_pred), 1))

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root mean squared error:", rmse.round(2)
```
## Model Error (RMSE ~ 40 Days)
**Root mean squared error: 39.75**

After playing around with the n_estimators that would give me the lowest Mean Squared Error, it appears that 91 n_estimators with a random state of 38 gives me the lowest Mean Squared Error with this random forest model.
__
## PyCaret LGBM Prediction

After I built the random forest model, I discovered PyCaret. I have now built a Light Gradient Boosted Model using this Python library.
```
PYCARET CITATION
@Manual{PyCaret,
  author  = {Moez Ali},
  title   = {PyCaret: An open source, low-code machine learning library in Python},
  year    = {2020},
  month   = {April},
  note    = {PyCaret version 1.0.0},
  url     = {https://www.pycaret.org}
}
```

My code:
```
import pandas as pd
from pycaret.regression import *

df = pd.read_csv('cleaned-patient_wait_times.csv')
df.head()

df.dtypes

cat_features = ['Province','Indicator','Year','Metric']

experiment = setup(df, target = 'Result', categorical_features = cat_features)

best_model = compare_models()
# Seems like the LGBM is the best model for me for this data

#Evaluate the model with different graphs showing its accuracy
#Fantastic part of PyCaret is the built in graphs

evaluate_model(best_model)
```

![Image1](https://github.com/Gavin-Thomas/Wait-Times-for-Priority-Procedures-in-Canada/blob/main/Images/Residuals_plot.png?raw=true)

_Figure 5_. Residuals plot of the Light-Gradient-Boosted Model. Observe how the accuracy of the model diminishes as the number of days increases.

![Image2](https://github.com/Gavin-Thomas/Wait-Times-for-Priority-Procedures-in-Canada/blob/main/Images/LGBM_residuals.png?raw=true)

![Image3](https://github.com/Gavin-Thomas/Wait-Times-for-Priority-Procedures-in-Canada/blob/main/Images/Feature_importance.png?raw=true)
_Figure 6_. Feature Importance plot for the top 10 features. 
---
