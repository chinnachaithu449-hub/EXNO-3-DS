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
~~~

 import pandas as pd 
df=pd.read_csv("Encoding Data.csv") 
df
~~~
<img width="492" height="362" alt="Screenshot 2026-08-18 152037" src="https://github.com/user-attachments/assets/0c86019f-9ee5-48cb-835a-ff81d3c36ba1" />

~~~
# ORDINAL ENCODING 
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder 
pm=['Hot','Warm','Cold'] 
e1=OrdinalEncoder(categories=[pm]) 
e1.fit_transform(df[["ord_2"]])
~~~


<img width="436" height="225" alt="Screenshot 2026-08-18 152248" src="https://github.com/user-attachments/assets/91558b87-b7d3-479d-8553-fb0db9f33b2a" />


~~~
df['bo2']=e1.fit_transform(df[["ord_2"]]) 
df
~~~
<img width="542" height="360" alt="Screenshot 2026-08-18 152433" src="https://github.com/user-attachments/assets/0299c9df-f853-4b78-b9ed-d6457f77d1aa" />

~~~
# Label Encoder ( orders in alphabetical order) 
le=LabelEncoder() 
dfc=df.copy() 
dfc['ord_2']=le.fit_transform(dfc['ord_2']) 
dfc
~~~
<img width="516" height="365" alt="Screenshot 2026-08-18 152538" src="https://github.com/user-attachments/assets/14b8109d-8fa6-4c2e-b219-5ccea9d75132" />

~~~
# ONE HOT ENCODING 
from sklearn.preprocessing import OneHotEncoder 
ohe=OneHotEncoder(sparse_output=False) 
df2=df.copy() 
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) # Orders in Alphabetical Order Blue , Green, Red 
df2=pd.concat([df2,enc],axis=1) 
df2
~~~
<img width="637" height="363" alt="Screenshot 2026-08-18 152641" src="https://github.com/user-attachments/assets/b56fa1a5-c3be-4464-aca5-fbe2765c2567" />

~~~
pd.get_dummies(df2,columns=["nom_0"])
~~~
<img width="863" height="366" alt="Screenshot 2026-08-18 152806" src="https://github.com/user-attachments/assets/d4c23370-4390-4aef-b9df-d7d5a769efdc" />

~~~
pip install --upgrade category_encoders
~~~
<img width="1242" height="480" alt="Screenshot 2026-08-18 152930" src="https://github.com/user-attachments/assets/3071c921-e2c3-4eab-9397-a63743a0a575" />

~~~
# BINARY ENCODER 
from category_encoders import BinaryEncoder 
df=pd.read_csv("data.csv") 
df
~~~
<img width="692" height="372" alt="Screenshot 2026-08-18 153047" src="https://github.com/user-attachments/assets/01ec9c2f-ba98-4e4e-a178-71d7a41282be" />

~~~
be=BinaryEncoder() 
nd=be.fit_transform(df['Ord_2'])  
dfb
~~~

<img width="703" height="360" alt="Screenshot 2026-08-18 153230" src="https://github.com/user-attachments/assets/0d0e9454-31c9-4447-9b16-79a0f3a4f80d" />

~~~
# MEAN ENCODING 
from category_encoders import TargetEncoder 
te=TargetEncoder() 
CC=df.copy() 
new=te.fit_transform(X=CC["City"],y=CC["Target"]) 
CC=pd.concat([CC,new],axis=1) 
CC
~~~
<img width="760" height="365" alt="Screenshot 2026-08-18 153358" src="https://github.com/user-attachments/assets/2d0d1fde-c144-4dfc-a06e-10bbc2c9b5ad" />


~~~
# FEATURE TRANSFORMATION 
import pandas as pd 
from scipy import stats 
import numpy as np 
df=pd.read_csv("Data_to_Transform.csv") 
df
~~~
<img width="1007" height="452" alt="Screenshot 2026-08-18 153520" src="https://github.com/user-attachments/assets/78823f7f-ebc3-4e8b-8e0c-b7898f0985b8" />

~~~
df.skew()
~~~
<img width="511" height="122" alt="Screenshot 2026-08-18 153609" src="https://github.com/user-attachments/assets/825a5074-5d2a-4991-83e2-9a0cdf381129" />


~~~
# 1. LOG TRANSFORMATION 
np.log(df["Highly Positive Skew"])
~~~
<img width="671" height="271" alt="Screenshot 2026-08-18 153705" src="https://github.com/user-attachments/assets/de847edc-e23c-4749-8f0b-2b98b7b28e4c" />

~~~
# 2. RECIPROCAL TRANSFORMATION 
np.reciprocal(df["Moderate Positive Skew"])
~~~
<img width="841" height="321" alt="Screenshot 2026-08-18 153759" src="https://github.com/user-attachments/assets/47cd28d6-616b-4c9c-ac23-e6c0815dfd8c" />

~~~
# 4. SQUARE ROOT TRANSFORMATION 
np.sqrt(df["Highly Positive Skew"])
~~~
<img width="750" height="278" alt="Screenshot 2026-08-18 153916" src="https://github.com/user-attachments/assets/0a5cb2c9-08ce-4a93-871c-7e9fdb70ebf8" />

~~~
# 5. SQUARE TRANSFORMATION 
np.square(df["Highly Positive Skew"])
~~~
<img width="667" height="280" alt="Screenshot 2026-08-18 154015" src="https://github.com/user-attachments/assets/abbf321f-f9f9-4877-ba1b-859f74ef0d70" />

~~~
# POWER TRANSFORMATIONS 
#        BOX COX 
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"]) 
df
~~~
<img width="1190" height="442" alt="Screenshot 2026-08-18 154113" src="https://github.com/user-attachments/assets/ea948694-64dc-4b5b-adc0-d08a8a81e9cf" />


~~~
df.skew()
~~~
<img width="587" height="160" alt="Screenshot 2026-08-18 154223" src="https://github.com/user-attachments/assets/82047452-528f-4229-a71f-0c14a513e081" />

~~~
 # YEO_JOHNSON 
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df
~~~
<img width="1330" height="385" alt="Screenshot 2026-08-18 154428" src="https://github.com/user-attachments/assets/6d0c2b30-6421-4bab-aebd-fa18d3ec87a0" />


~~~
# QUANTILE TRANSFORMATION 
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal') 
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
df
~~~
<img width="1337" height="468" alt="Screenshot 2026-08-18 154516" src="https://github.com/user-attachments/assets/d4e5cb57-d47b-474b-a9e3-16343a337a75" />


~~~
import seaborn as sns 
import statsmodels.api as sm 
# STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION 
import matplotlib.pyplot as plt 
sm.qqplot(df["Moderate Negative Skew"],line='45') # QQ - QUANTILE QUANTILE PLOT plt.show()
~~~
<img width="928" height="555" alt="Screenshot 2026-08-18 154632" src="https://github.com/user-attachments/assets/e5b07fc7-36f3-42e8-a5ef-c5783187d039" />

~~~
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
~~~
<img width="1036" height="740" alt="Screenshot 2026-08-18 154746" src="https://github.com/user-attachments/assets/b7836950-5c86-43ce-a7cc-8f978e203b3b" />

~~~
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal', n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"], line='45')
plt.show()

~~~
<img width="973" height="585" alt="Screenshot 2026-08-18 154912" src="https://github.com/user-attachments/assets/de3b5bcd-e0e4-4a8d-8c26-66648e5dfacf" />







# RESULT:
      The dataset was successfully cleaned, encoded, transformed, and saved as a preprocessed dataset. This makes the data suitable for further machine learning and data analysis tasks.

       
