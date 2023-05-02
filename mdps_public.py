import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import matplotlib as plt
import seaborn as sns

# loading the saved models

heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Main Menu',
                          ['About the Project', 'Heart Disease Prediction',  'Attribute Description', 'Analysis and Visualization', 'Findings and Outcomes'],
                          icons=['heart'],
                          default_index=0)
    
df = pd.read_csv('data.csv')

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

        cp_options = ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic']
        cp = st.selectbox('Chest Pain type', options=cp_options)
        cp = 0 if cp == 'Typical Angina' else (1 if cp == 'Atypical Angina' else (2 if cp == 'Non-Anginal Pain' else 3))

        chol = st.text_input('Serum Cholesterol in mg/dl', value = 200)
        
        oldpeak = st.number_input('ST depression induced by exercise', value=0.0, step=0.1)

        thal_options = ['Normal', 'Fixed Defect', 'Reversible Defect']
        thall = st.selectbox('Thal', options=thal_options)
        thall = 0 if thall == 'Normal' else (1 if thall == 'Fixed Defect' else 2)

        
    with col2:

        trtbps = st.slider('Resting Blood Pressure', 80, 200, 120)
        
        sex_options = ['Male', 'Female']
        sex = st.selectbox('Sex', options=sex_options)
        sex = 0 if sex == 'Female' else 1
              
        restecg_options = [0, 1, 2]
        restecg_labels = ['Normal', 'Abnormality in ST-T wave', 'Showing probable or definite left ventricular hypertrophy']
        restecg = st.selectbox('Resting Electrocardiographic Results', options=restecg_options, format_func=lambda x: restecg_labels[x])

        slope_options = [0, 1, 2]
        slp = st.selectbox('Slope of the peak exercise ST segment', options=slope_options)
    
    with col3:

        thalachh = st.slider('Maximum Heart Rate achieved', min_value=60, max_value=220, value=120)
        
        fbs_options = ['False', 'True']
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=fbs_options)
        fbs = 0 if fbs == '< 120' else 1

        exang_options = ['No', 'Yes']
        exng = st.selectbox('Exercise induced angina', options=exang_options)
        exng = 0 if exng == 'No' else 1
    
        caa_options = ['0', '1', '2', '3', '4']
        caa = st.selectbox('Major Vessels Colored by Fluoroscopy', options=caa_options)

        
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

elif (selected == 'Attribute Description'):

    def attribute_description():
        st.markdown("# Attribute Description")
        st.markdown("The following are the attributes used in the heart disease dataset:")
    
        st.markdown("1. **age**: The person's age in years.")
        st.markdown("2. **sex**: The person's gender (1 = male, 0 = female).")
        st.markdown("3. **cp**: Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic).")
        st.markdown("4. **trestbps**: Resting blood pressure (mm Hg) when the person was admitted to the hospital.")
        st.markdown("5. **chol**: Serum cholesterol (mg/dl) level.")
        st.markdown("6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).")
        st.markdown("7. **restecg**: Resting electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy).")
        st.markdown("8. **thalach**: Maximum heart rate achieved during the exercise test.")
        st.markdown("9. **exang**: Exercise induced angina (1 = yes, 0 = no).")
        st.markdown("10. **oldpeak**: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot.)")
        st.markdown("11. **slope**: The slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping).")
        st.markdown("12. **ca**: The number of major vessels (0-4).")
        st.markdown("13. **thal**: A blood disorder called thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect).")

    attribute_description()

elif (selected == 'Analysis and Visualization'):

    st.header("Data Exploration")

    # Print first five rows
    st.write("### Printing first five rows:")
    st.write(df.head())

    # Statistical data
    st.write("### Checking Statistical Data:")
    st.write(df.describe())

    # Check for datatypes and attributes
    st.write("### Checking for datatypes and attributes:")
    st.write(df.info())

    # Check for null values
    st.write("### Checking for null values:")
    st.write(df.isnull().sum())

    # Print the columns
    st.write("### Printing the columns:")
    st.write(df.columns)

    # Check for duplicate rows
    st.write("### Checking for duplicate rows:")
    st.write(df.duplicated().sum())

    # Removing duplicates
    df.drop_duplicates(inplace=True)

    # Check the shape
    st.write("### Checking the shape:")
    st.write(f"Number of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

    # Computing the correlation matrix
    st.write("### Computing the correlation matrix:")
    st.write(df.corr())

    # Data Visualization
    st.header("Data Visualization")
    
    # Breakdown of Gender
    st.write("### Breakdown of Gender:")
    x = (df.sex.value_counts())
    st.write(f"Number of people having gender as Female are {x[0]} and Number of people having gender as Male are {x[1]}")
    p = sns.countplot(data=df, x="sex")
    plt.show()
    st.pyplot(p.figure)

    # Breakdown for chest pain
    st.write("### Breakdown for Chest Pain:")
    x = (df.cp.value_counts())
    st.write("The frequency of each type of chest pain:")
    st.write(f"Typical Angina - {x[0]}")
    st.write(f"Atypical Angina - {x[1]}")
    st.write(f"Non-Anginal pain - {x[2]}")
    st.write(f"Asymptomatic - {x[3]}")
    p = sns.countplot(data=df, x="cp")
    plt.show()
    st.pyplot(p.figure)

    # Breakdown for Fasting blood sugar
    st.write("### Breakdown for Fasting Blood Sugar:")
    x = (df.fbs.value_counts())
    st.write(f"Cases where fasting blood sugar < 120 mg/dl: {x[0]}")
    st.write(f"Cases where fasting blood sugar > 120 mg/dl: {x[1]}")
    p = sns.countplot(data=df, x="fbs")
    plt.show()
    st.pyplot(p.figure)

    # Breakdown of ECG
    st.write("### Breakdown of ECG Results:")
    x = (df.restecg.value_counts())
    st.write("Resting ECG results:")
    st.write(x)
    p = sns.countplot(data=df, x="restecg")
    plt.show()
    st.pyplot(p.figure)

    st.write("### Exercise Induced Angina Breakdown")
    x = df.exng.value_counts()
    st.write(f"Number of Exercise induced angina:")
    st.write(x)
    p = sns.countplot(data=df, x="exng")
    plt.show()
    st.pyplot(p.figure)
    
    st.write("### Thalium Stress Test Breakdown")
    x = df.thall.value_counts()
    st.write("Thall Count is min for type 0 and max for type 2:\n", x)
    p = sns.countplot(data=df, x="thall")
    st.pyplot(p.figure)
    
    st.write("### Density distribution for Age")
    p = sns.displot(df.age, color="red", label="Age", kde= True)
    st.pyplot(p)
    st.write("Density distribution is highest for age group 55 to 60")
    
    st.write("### Density Distribution for Resting Blood Pressure")
    p = sns.displot(df.trtbps, color="green", label="Resting Blood Pressure", kde= True)
    st.pyplot(p)
    st.write("Trtbs has the highest count around 130")
    
    st.write("### Heart Attack Vs Age")
    plt.figure(figsize=(10,10))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.distplot(df[df['output'] == 0]["age"], color='green',kde=True,) 
    sns.distplot(df[df['output'] == 1]["age"], color='red',kde=True)
    st.pyplot()

    st.write("### Cholesterol versus chances of heart disease")
    plt.figure(figsize=(10,10))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.distplot(df[df['output'] == 0]["chol"], color='green',kde=True,) 
    sns.distplot(df[df['output'] == 1]["chol"], color='red',kde=True)
    st.pyplot()

    st.write("### Resting blood pressure vs chances of heart disease")
    plt.figure(figsize=(10,10))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.distplot(df[df['output'] == 0]["trtbps"], color='green',kde=True,) 
    sns.distplot(df[df['output'] == 1]["trtbps"], color='red',kde=True)
    st.pyplot()

    st.write("### Heart Rate versus chances of heart disease")
    plt.figure(figsize=(10,10))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.distplot(df[df['output'] == 0]["thalachh"], color='green',kde=True,) 
    sns.distplot(df[df['output'] == 1]["thalachh"], color='red',kde=True)
    st.pyplot()
















