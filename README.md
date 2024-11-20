# EX-06 ANALYZING A DATASET WITH VARIOUS STAGES OF DATA SCIENCE
### AIM:
To Analyze a data set with Various stages of data science. &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;**DATE :19-11-2024**
### ALGORITHM:
Step 1: Include the necessary python Library.<BR>
Step 2: Choose your own dataset and read it.<BR>
Step 3: Implement Data analysis using the necessary columns.<BR>
Step 4: Perform Data Preprocessing steps for the necessary columns.<BR>
Step 5: Perform Feature Engineering process for the categorical columns.<BR>
Step 6: Implement Advanced data Visualization for the columns necessary.<BR>
### PROGRAM:
**Developed By: ROSELIN MARY JOVITA S - 212222230122**
##### Importing Libraries
```Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
```
##### Reading DataSet
```Python
dt=pd.read_csv("/content/titanic_dataset.csv")
dt
```
![Screenshot 2024-11-20 205152](https://github.com/user-attachments/assets/650318fb-1169-4eb4-835b-8e98ba1f5246)


##### Data Analysis
```Python
df.isnull().sum()
df.info()
```
![Screenshot 2024-11-20 205200](https://github.com/user-attachments/assets/3d530166-f00b-41d9-8d9d-bafdb4ecb505)


##### Data Preprocessing
```Python
for column in ['Age', 'Fare']:
  dt[column].fillna(dt[column].mean(), inplace=True)
for column in ['Embarked', 'Cabin']:
  dt[column].fillna(dt[column].mode()[0], inplace=True)
dt.replace('?', np.nan, inplace=True)
print(dt.isnull().sum())
```
![Screenshot 2024-11-20 205208](https://github.com/user-attachments/assets/e6e7103d-460d-410f-8eaf-611b0e70d5a4)


##### Feature Engineering
```Python
scl=MinMaxScaler()
dt[['Age']]=scl.fit_transform(dt[['Age']])

le=LabelEncoder()
dt['Fare']=le.fit_transform(dt['Fare'])

dt.head(5)
```
![Screenshot 2024-11-20 205229](https://github.com/user-attachments/assets/354f6e8c-7b42-48b3-8a62-5520f0012661)

##### Data Visualization
CORRELATION MATRIX
```Python
corr = dt.select_dtypes(include=np.number).corr()  # Select only numeric columns
sns.heatmap(corr, annot=True)
```

![Screenshot 2024-11-20 205243](https://github.com/user-attachments/assets/6267488b-9ede-4bfd-a0cc-13b25b5154bc)

PAIR PLOT

```
sns.pairplot(dt)
```

![Screenshot 2024-11-20 205312](https://github.com/user-attachments/assets/e5f0d893-e091-4032-969c-b2966da061eb)


### RESULT:
Thus, the analyzing the dataset with various stages of data science is implemented successfully.


