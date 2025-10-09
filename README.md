## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
<img width="507" height="505" alt="image" src="https://github.com/user-attachments/assets/9f649ac3-d478-4dfb-b193-6f204eec7b26" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="849" height="357" alt="image" src="https://github.com/user-attachments/assets/f0f4ae8a-743a-47d7-8164-933e5fb8ab74" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="623" height="457" alt="image" src="https://github.com/user-attachments/assets/74d87ef5-0f78-4cc5-8be9-4d1be7071e0d" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="596" height="524" alt="image" src="https://github.com/user-attachments/assets/a7dc0363-29b9-46af-ac15-8d4e565eb7f4" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) # Orders in Alphabetical Order Blue , Green, Red
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="1391" height="592" alt="image" src="https://github.com/user-attachments/assets/e2686866-8230-4030-881a-f03b73c90b9d" />

```
 pd.get_dummies(df2,columns=["nom_0"])
```
<img width="995" height="465" alt="image" src="https://github.com/user-attachments/assets/cfa189a3-803e-4ee2-8b3a-737557138ee1" />

```
pip install --upgrade category_encoders
```
<img width="1474" height="579" alt="image" src="https://github.com/user-attachments/assets/c373b393-62df-4af9-b279-e8039cad678d" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
<img width="679" height="497" alt="image" src="https://github.com/user-attachments/assets/9bf687b3-1527-4ab0-b34f-31cf51a9691a" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb
```
<img width="1046" height="527" alt="image" src="https://github.com/user-attachments/assets/a5434180-225a-42e4-b5d8-56195170bed6" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
<img width="1069" height="623" alt="image" src="https://github.com/user-attachments/assets/ab13ed18-0ca5-49f5-b67e-5805c5f0c246" />

```
df.skew()
```
<img width="532" height="175" alt="image" src="https://github.com/user-attachments/assets/77599c02-fe40-4208-b254-b0ef7f651fac" />

```
np.log(df["Highly Positive Skew"])
```
<img width="832" height="343" alt="image" src="https://github.com/user-attachments/assets/c8768441-81b9-4ff7-bd65-e7255a2c7bc7" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="766" height="330" alt="image" src="https://github.com/user-attachments/assets/9b3fa134-be39-4fb6-979d-94ef0f9a9b5c" />

```
 np.sqrt(df["Highly Positive Skew"])
```
<img width="755" height="336" alt="image" src="https://github.com/user-attachments/assets/06d0337c-a692-4284-b163-de2e85b60bf5" />

```
np.square(df["Highly Positive Skew"])
```
<img width="824" height="338" alt="image" src="https://github.com/user-attachments/assets/803a295c-2128-4d8a-90b8-ca071e7c0081" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1238" height="567" alt="image" src="https://github.com/user-attachments/assets/fb749ecc-c328-4126-92af-13279a2d56c0" />

```
df.skew()
```
<img width="644" height="197" alt="image" src="https://github.com/user-attachments/assets/52685388-305e-44c7-bef1-08d1eecdf702" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="1214" height="245" alt="image" src="https://github.com/user-attachments/assets/5aad7b61-a9d8-46e5-ac95-916cc0d244c7" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
<img width="1493" height="640" alt="image" src="https://github.com/user-attachments/assets/9350867f-3e18-4083-a395-32d53627ae1d" />

```
import seaborn as sns
import statsmodels.api as sm # STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45') # QQ - QUANTILE QUANTILE PLOT
plt.show()
```
<img width="1138" height="727" alt="image" src="https://github.com/user-attachments/assets/1198aed8-3e9f-42f8-bebe-69c01ccd1812" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL
plt.show()
```
<img width="1215" height="681" alt="image" src="https://github.com/user-attachments/assets/28bdc838-adce-4f37-8767-690488b3acca" />

```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="1102" height="713" alt="image" src="https://github.com/user-attachments/assets/064167fc-f976-4540-b751-de6a2f9f580e" />

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
<img width="1314" height="165" alt="image" src="https://github.com/user-attachments/assets/be629010-1993-46f2-a8cf-5cd8f56fe0d4" />

```
!pip install skfeature-chappers
```
<img width="1515" height="666" alt="image" src="https://github.com/user-attachments/assets/72d22288-35fe-4da6-9390-5bb828bd2521" />

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
<img width="1107" height="558" alt="image" src="https://github.com/user-attachments/assets/7269d8e1-94eb-4e2b-9063-1c60372efd45" />

```
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
<img width="1346" height="147" alt="image" src="https://github.com/user-attachments/assets/dd25126a-7551-402f-96d9-c9ae7b3d843e" />

```
df[categorical_columns]
```
<img width="1191" height="558" alt="image" src="https://github.com/user-attachments/assets/7e3992a6-3d3e-461a-9ce7-959015397daa" />




# RESULT:
Thus the given data and perform Feature Encoding and Transformation process is done.
