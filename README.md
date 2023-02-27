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


Everything I deleted or changed:
- Columns "Region", "Reporting Level", and "Unit of Measurement" (I did this one later on)
- Rows containing "Canada", 
- Changed all Provinces to their abbreviation (ex. AB instead of Alberta)



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



---
## Data Visualization 
---
## Random Forest Regressor
---
## Conclusion
