# <p align="center">Canadian Healthcare Analysis: Wait Times for Priority Procedures</p>
### <p align="center">By Gavin Thomas</p>
_____

<p align="left">Dataset: Canadian Institute for Health Information (CIHI). Wait Times for Priority Procedures in Canada — Data Tables. Ottawa, ON: CIHI; 2022.</p>

Dataset [HERE](https://www.cihi.ca/sites/default/files/document/wait-times-priority-procedures-in-canada-data-tables-en.xlsx)

_____

## Intro

Contained in this repository is my data analysis of the 2023 CIHI wait times data table for priority procedures. This CIHI dataset does not contain ALL types of priority procedures; rather, the data table contains a few different types of procedures that may give a general idea of waiting times depending on the procedure type, and the province of residence. 

The reason I chose this dataset is two-fold:
1. The current Canadian health-care system is known to have long wait times.
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
- Deleted all rows with proportion as a "Unit of Measurement"
- Deleted all Data years past 2019 (to not include COVID lockdown)
- Deleted all Data years with letters (ie. 2019FY)
- Changed All "Indicator Results" Column "Hours" to "Days" by dividing by 24.
- Changed Column Title "Province/Territory" to Province
- Changed Column Title "Data year" to "Year
- Changed Column Title "Indicator Result" to "Result"

Cleaned CSV file [HERE](https://github.com/Gavin-Thomas/Wait-Times-for-Priority-Procedures-in-Canada/blob/main/cleaned-patient_wait_times.csv)

I also (manually - because it was easy) deleted the columns that I didn't want. These were "Region" & "Reporting Level"

I used pandas and numpy for data cleaning.

```
import pandas as pd
import numpy as np

# I then read the CSV using pandas, and got a header of the file

df = pd.read_csv("patient_wait_times.csv")
df.head(5)
```
The file did not look pretty at this point, so I dropped all N/A values.
```
df = df.dropna()
```


---
## Data Visualization 
---
## Random Forest Regressor
---
## Conclusion
