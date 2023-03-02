# <p align="center">(DRAFT)Canadian Healthcare Analysis: Wait Times for Priority Procedures</p>
### <p align="center">By Gavin Thomas</p>
_____

<p align="left">Dataset: Canadian Institute for Health Information (CIHI). Wait Times for Priority Procedures in Canada — Data Tables. Ottawa, ON: CIHI; 2022.</p>

Dataset [HERE](https://www.cihi.ca/sites/default/files/document/wait-times-priority-procedures-in-canada-data-tables-en.xlsx)

SKILLS DEMONSTRATED:
- Data cleaning with Pandas
- Data visualization with Seaborn and Matplotlib
- Machine learning with Scikit-Learn
_____

## Intro

Contained in this repository is my data analysis of the 2023 CIHI wait times data table for priority procedures. This CIHI dataset does not contain ALL types of priority procedures; rather, the data table contains a few different types of procedures that may give a general idea of waiting times depending on the procedure type, and the province of residence. 

The reason I chose this dataset is two-fold:
1. The Canadian health-care system is known to have long wait times.
2. The CIHI has released a clear, well-formatted dataset.

When I did my data analysis I did not include all parameters. The reason I did this was to simplify my life, and to reduce any confounders to the random forest regressor I built, as well as to generate clear graphs.

In this readme file, I will take you through my process of data cleaning, analysis/visualization, and model building. I have also attached my code for data cleaning, the cleaned csv file, my code for data visualization, and my code for the random forest regressor.

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
- Deleted all Data years past 2019 (to not include COVID lockdown)
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
It appears that on average, Prince Edward Island has the shortest wait times and Nova Scotia has the longest wait times. Let's visualize the data to get a better understanding.

For this next part, I use Seaborn which is based off MatPlotLib. Seaborn has some fantastic tools. I recently started using violin plots more, and I believe they give a fantastic estimate of 
```
sb.set(rc={'figure.figsize':(25,10)})

violin_plot = sb.violinplot(y="Result",x="Province",data=df)
violin_plot.set_xlabel('Indicator', fontsize=40, fontfamily = 'Times New Roman')
violin_plot.set_ylabel('Result (Days)', fontsize=40, fontfamily = 'Times New Roman')
violin_plot.tick_params(axis='both', which='major', labelsize=25, width=2, length=6, pad=8)
```
![ViolinPlot-Province](https://github.com/Gavin-Thomas/Wait-Times-for-Priority-Procedures-in-Canada/blob/main/Images/ViolinPlot-Province.png?raw=true)
_Figure 1_. Violin plot of wait time in days for each Canadian province (excluding territories) for priority procedures. 

EXPLANATION OF FIGURE 1 HERE

![BarPlot-Provice](https://github.com/Gavin-Thomas/Wait-Times-for-Priority-Procedures-in-Canada/blob/main/Images/BarGraph.png?raw=true)
_Figure 2_. Bar plot of wait time in days for each Canadian province (excluding territories) for priority procedures. SEM bars included.


---
## Random Forest Regressor
---
## Conclusion
