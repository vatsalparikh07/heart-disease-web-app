import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models

heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Main Menu',
                          ['About the Project', 'Heart Disease Prediction',  'Attributes Description', 'Analysis and Visualization', 'Findings and Outcomes'],
                          icons=['heart'],
                          default_index=0)

if (selected == 'About the Project'):

    st.title("Project Description")

    st.markdown("""
    Cardiovascular diseases (CVDs) are the leading cause of death globally, taking an estimated 17.5 million lives each year, accounting for nearly 31 percent of global deaths. It is important to identify those at the highest risks of CVDs to ensure appropriate treatment and preventing premature deaths.
    
    This project is a web application that predicts the presence of heart disease in a patient based on various medical factors such as age, blood pressure, cholesterol level, etc. The project uses machine learning algorithms to analyze the data and make predictions.

    ### Data

    The data used in this project is the [Heart Disease UCI dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) from the UCI Machine Learning Repository. The dataset contains 303 samples with 14 features, including age, sex, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar level, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise relative to rest, slope of the peak exercise ST segment, number of major vessels colored by fluoroscopy, and thalassemia.
    
    ### Machine Learning Algorithm

    The machine learning algorithm used for this project is [Support Vector Machine](https://scikit-learn.org/stable/modules/svm.html). The model is able to predict whether a patient has heart disease with an accuracy of 93.44%.

    ### Libraries Used

    This project uses the following libraries:

    * Pandas
    * NumPy
    * Scikit-learn
    * Matplotlib
    * Seaborn
    * Streamlit

    ### About the Developer

    This project was developed by Vatsal Parikh. You can find more of my projects on [GitHub](https://github.com/vatsalparikh07).

    """)

# Heart Disease Prediction Page
elif (selected == 'Heart Disease Prediction'):
    # page title
    st.title('Heart Disease Prediction using ML')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:

        age = st.slider('Age', 1, 100, 25)

        trtbps = st.slider('Resting Blood Pressure', 80, 200, 120)

        thalachh = st.slider('Maximum Heart Rate achieved', min_value=60, max_value=220, value=120)

        oldpeak = st.number_input('ST depression induced by exercise', value=0.0, step=0.1)

        thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']
        thall = st.selectbox('Thal', options=thal_options)
        thall = 0 if thall == 'Normal' else (1 if thall == 'Fixed Defect' else 2)

        
    with col2:

        sex_options = ['Male', 'Female']
        sex = st.selectbox('Sex', options=sex_options)
        sex = 0 if sex == 'Female' else 1

        chol = st.text_input('Serum Cholesterol in mg/dl')

        restecg_options = [0, 1, 2]
        restecg_labels = ['Normal', 'Abnormality in ST-T wave', 'Showing probable or definite left ventricular hypertrophy']
        restecg = st.selectbox('Resting Electrocardiographic Results', options=restecg_options, format_func=lambda x: restecg_labels[x])

        slope_options = [0, 1, 2]
        slp = st.selectbox('Slope of the peak exercise ST segment', options=slope_options)
    
    with col3:

        cp_options = ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic']
        cp = st.selectbox('Chest Pain type', options=cp_options)
        cp = 0 if cp == 'Typical Angina' else (1 if cp == 'Atypical Angina' else (2 if cp == 'Non-Anginal Pain' else 3))

        fbs_options = ['False', 'True']
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=fbs_options)
        fbs = 0 if fbs == '< 120' else 1

        exang_options = ['No', 'Yes']
        exng = st.selectbox('Exercise induced angina', options=exang_options)
        exng = 0 if exng == 'No' else 1
    
        caa_options = ['0', '1', '2', '3', '4']
        caa = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', options=caa_options)

        
    # code for Prediction

    heart_diagnosis = ''
    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        heart_prediction = heart_disease_model.predict([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]])                          
        
        if (heart_prediction[0] == 1):
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

elif (selected == 'Attributes Description'):
    st.set_page_config(page_title="Attributes of Heart Disease Prediction", page_icon=":heart:")

    # Define page header
    st.markdown("# Attributes of Heart Disease Prediction")















