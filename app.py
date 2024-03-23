import streamlit as st
import pandas as pd
import numpy as np
import pandas_profiling

from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import setup as setup_reg
from pycaret.regression import compare_models as compare_models_reg
from pycaret.regression import save_model as save_model_reg
from pycaret.regression import plot_model as plot_model_reg
from pydantic.v1 import BaseModel
from pydantic.v1.utils import lenient_isinstance

from pycaret.classification import setup as setup_class
from pycaret.classification import compare_models as compare_models_class
from pycaret.classification import save_model as save_model_class
from pycaret.classification import plot_model as plot_model_class




url="https://www.linkedin.com/in/hakizimana-jean-claude-714195163"

@st.cache
def load_data(file):
    data=pd.read_csv(file)
    return data

def main():
    st.title("AUTOMATIQUE MACHINE LEARNING EDA DATA ANALYSIS FOR REGRESSION AND CLASSIFICATION")
    st.sidebar.write("[Author: Jean Claude HAKIZIMANA](%s)"%url)
    st.sidebar.markdown(
        "**This web app is a no code tool for EDA and building ML model for Classification and Regression"
        "1. Load your dataset file in CSV format; \n"
        "2. Click on *profile Dataset* button to generate the pandas profiling of the dataset; \n"
        "3. Choose your target column; \n"
        "4. Choose the machine learning task (Regression or Classification); \n"
        "5. Click Run Modeling to start the training process. \n"
        "When the model is built, you can view the result like the pipeline model, residuals plot, ROC Curve, Confusion Matrix,..."
        "\n6. Download the model to your computer."        
    )
    file = st.file_uploader("Upload your dataset in csv format", type=["csv"])
    
    if file is not None:
        data = load_data(file)
        st.dataframe(data.head())
        
        profile = st.button("Profile Dataset")
        if profile:
            profile_df = data.profile_report()
            st_profile_report(profile_df)
        
        target = st.selectbox("Select the target variable", data.columns)
        task = st.selectbox("Select a ML task", ["Regression", "Classification"])
        data = data.dropna(subset=[target])
        
        if task == "Regression":
            if st.button("Run Modelling"):
                exo_reg = setup_reg(data, target=target)
                model_reg = compare_models_reg()
                save_model_reg(model_reg, "best_reg_model")
                st.success("Regression Model built Successfully!!!")
                
                # Results
                st.write("Residuals")
                plot_model_reg(model_reg, plot='residuals', save=True)
                st.image("Residuals.png")
                
                st.write("Feature Importance")
                plot_model_reg(model_reg, plot='feature', save=True)
                st.image("Feature Importance.png")
                
                with open('best_reg_model.pkl', 'rb') as f:
                    st.download_button('Download Pipeline Model', f, file_name="best_reg_model.pkl")
                    
        if task == "Classification":
            if st.button("Run Modelling"):
                exp_class = setup_class(data, target=target)
                model_class = compare_models_class()
                save_model_class(model_class, "best_class_model")
                st.success("Classification Model built Successfully!!!")
                
                # Results
                col5, col6 = st.columns(2)
                with col5:
                    st.write("ROC Curve")
                    plot_model_class(model_class, save=True)
                    st.image("AUC.png")
                     
                with col6:
                    st.write("Classification Report")
                    plot_model_class(model_class, plot='class_report', save=True)
                    st.image("Class Report.png")
                     
                col7, col8 = st.columns(2)
                with col7:
                    st.write("Confusion Matrix")
                    plot_model_class(model_class, plot='confusion_matrix', save=True)
                    st.image("Confusion Matrix.png")
                
                with col8:
                    st.write("Feature Importance")
                    plot_model_class(model_class, plot='feature', save=True)
                    st.image("Feature Importance.png")
                
                with open('best_class_model.pkl', 'rb') as f:
                    st.download_button('Download Pipeline Model', f, file_name="best_class_model.pkl")
    
    else:
        st.image(r"C:\Users\alvin\Pictures\Screenshots\buish_buja.png")
    
if __name__ == '__main__':
    main()

