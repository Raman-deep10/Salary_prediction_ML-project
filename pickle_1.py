# Decision tree regressor
from matplotlib.backend_bases import LocationEvent
import pandas as pd
import numpy as np

np.random.seed(42)
n_samples= 1000
years_exp= np.random.randint(1,16)
level=np.random.randint(1,4,size=n_samples)
loaction_chc=["Banglore","Pune","Delhi","Chennai"]
location=np.random.choice(loaction_chc,size= n_samples)

current_ctc=np.round(np.random.uniform(3,15,size=n_samples),2)
expected_ctc= np.round(current_ctc +0.5+years_exp+8.8+level+np.random.normal(0,0.5,n_samples),2)

df=pd.DataFrame(
    {"Years of experience": years_exp,
     "Level":level,
     "Location": location ,
     "Current_ctc":current_ctc,
     "Expected_ctc": expected_ctc
}
)
x=df.drop("Expected_ctc",axis=1)
y=df["Expected_ctc"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import pickle


num_fet=["Years of experience","Level","Current_ctc"]
num_trn=StandardScaler()
cat_fet= ["Location"]
cat_trn= OneHotEncoder(handle_unknown="ignore")

preprocessor= ColumnTransformer(
    transformers=[
        ('num',num_trn, num_fet),
        ('cat',cat_trn,cat_fet)
    ])

model=Pipeline([
    ("preprocessor",preprocessor),
    ("regressor" , DecisionTreeRegressor(max_depth=5,random_state=42))
])

model.fit(x_train,y_train)
y_pred= model.predict(x_test)

print(r2_score(y_test,y_pred))


# pickle model 
with open("salprd.pkl","wb") as fp:
    pickle.dump(model,fp)

















