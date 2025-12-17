import pickle
import streamlit as st
import pandas as pd

with open("salprd.pkl","rb") as fp:
    # model=pickle.load(fp)
    
    model = pickle.load(open("salprd.pkl","rb"))

st.header("Expected Salary predictor")
years= st.slider("Years of Experience",1,20,5)
level_input= st.selectbox("Level",options=[1,2,3],
                          format_func= lambda x:{1:"Junior",2:"Mid",3:"Expert"}[x])

current_ctc_input= st.number_input("Current CTC (in lakhs)",0.0,15.0,5.0,1.0)
loaction_chc=["Banglore","Pune","Delhi","Chennai"]
loc= st.selectbox("loaction",loaction_chc)

if st.button("Predict_salary"):
    new_row= pd.DataFrame(
       {"Years of experience": [years],
     "Level":[level_input],
     "Location": [loc] ,
     "Current_ctc":[current_ctc_input],
    } 
     )

    prd= model.predict(new_row)[0]
    st.success(f"Expected_ctc{prd: .2f}Lakhs")