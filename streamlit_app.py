pip install sklearn
import streamlit as st
import numpy as np
import pandas as pd
from  import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from imblearn.over_sampling import SMOTE
# import joblib  # For saving and loading models.  NOT USED but kept for potential expansion - Removed this as it is not used


# --- Load Dataset ---
@st.cache_data()  # Use st.cache_data to cache data loading for performance
def load_data():
    """Loads the dataset.  Cached for performance."""
    try:
        tdata = pd.read_csv("/content/Integrated.csv", header=None, na_values=[-9])
        return tdata
    except FileNotFoundError:
        st.error(
            "Error: 'Integrated.csv' not found.  Please make sure the data file is in the same directory as the script, or provide the correct path."
        )
        st.stop()  # Stop execution if the file is not found
    except Exception as e:
        st.error(f"Error reading data: {e}")
        st.stop()  # Stop on other errors as well
    return None  # Important:  Return None in case of other errors


# --- Model Training ---
@st.cache_resource  # Use st.cache_resource for caching models
def train_models(training_X, training_y):
    """Trains the machine learning models.  This function is called once."""
    st.info("Training models... (this may take a moment)")  # Added progress indication

    svm_model = svm.SVC(kernel='rbf', C=5, gamma='scale', probability=True)
    svm_model.fit(training_X, training_y)

    log_reg = LogisticRegression(solver='lbfgs')
    log_reg.fit(training_X, training_y)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(training_X, training_y)

    nn_model = Sequential(
        [
            Dense(
                10,
                activation='relu',
                input_dim=10,
                kernel_regularizer=regularizers.l2(0.01),
            ),
            Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            Dense(1, activation='sigmoid'),  # Binary Classification: Sigmoid Activation
        ]
    )
    nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    nn_model.fit(training_X, training_y, epochs=50, verbose=0)

    models = {  # Store the trained models
        'svm': svm_model,
        'log_reg': log_reg,
        'knn': knn,
        'nn': nn_model,
    }
    st.success("Models trained successfully!")  # Add success message
    return models  # Return the models


# --- Streamlit App ---
st.title("Heart Attack Risk Prediction")

# Load the data
tdata = load_data()  # Load the data

if tdata is not None:  # Only proceed if the data was loaded successfully
    # Select necessary columns
    new_data = tdata[[2, 3, 8, 9, 14, 15, 16, 17, 18, 31, 57]].copy()
    new_data.columns = [
        "Age",
        "Sex",
        "Chest Pain",
        "Blood Pressure",
        "Smoking Years",
        "Fasting Blood Sugar",
        "Diabetes History",
        "Family history Cornory",
        "ECG",
        "Pulse Rate",
        "Target",
    ]

    # Fill missing values
    new_data.fillna(new_data.mean(), inplace=True)
    new_data["Diabetes History"].fillna(
        new_data["Diabetes History"].mode()[0], inplace=True
    )
    new_data["Family history Cornory"].fillna(
        new_data["Family history Cornory"].mode()[0], inplace=True
    )

    # Convert target to binary (Heart Attack Risk: 1 = High, 0 = Low)
    new_data["Target"] = new_data["Target"].apply(lambda x: 1 if x >= 3 else 0)

    # Split features and labels
    X = new_data.iloc[:, :-1].values
    y = new_data.iloc[:, -1].values

    # Normalize only continuous numerical features
    scaler = preprocessing.StandardScaler()
    X[:, [0, 3, 4, 5, 8, 9]] = scaler.fit_transform(X[:, [0, 3, 4, 5, 8, 9]])

    # Handle Class Imbalance Using SMOTE
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    # Split into training and testing sets
    training_X, testing_X, training_y, testing_y = train_test_split(
        X, y, test_size=0.10, random_state=70
    )

    # --- Initialize Session State ---
    if 'models' not in st.session_state:
        st.session_state.models = {}
        st.session_state.trained = False
        st.session_state.scaler = scaler  # Store the scaler
    else:
        scaler = st.session_state.scaler # Important:  Load scaler if models are in session state


    # --- Train Models ---
    if not st.session_state.trained:
        st.session_state.models = train_models(training_X, training_y)
        st.session_state.trained = True  # Set to true after training

    models = st.session_state.models  # Get models from session state
    #scaler = st.session_state.scaler  # get scaler from session state - Removed, scaler is loaded in init and else clause


    # Input fields
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Sex", options=["Male", "Female"])
    chest_pain = st.slider("Chest Pain Level (0-4)", min_value=0, max_value=4, value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
    smoking_years = st.number_input("Years of Smoking", min_value=0, max_value=50, value=0)
    fasting_blood_sugar = st.number_input("Fasting Blood Sugar", min_value=50, max_value=200, value=100)
    diabetes_history = st.selectbox("Diabetes History", options=["No", "Yes"])
    family_history = st.selectbox("Family History of Heart Disease", options=["No", "Yes"])
    ecg = st.number_input("ECG", min_value=-10.0, max_value=10.0, value=0.0)
    pulse_rate = st.number_input("Pulse Rate", min_value=50, max_value=150, value=70)

    if st.button("Predict Risk"):
        # Prepare user data
        user_data = np.array(
            [
                [
                    age,
                    1 if sex == "Male" else 0,
                    chest_pain,
                    blood_pressure,
                    smoking_years,
                    fasting_blood_sugar,
                    1 if diabetes_history == "Yes" else 0,
                    1 if family_history == "Yes" else 0,
                    ecg,
                    pulse_rate,
                ]
            ]
        )

        # Use the scaler that was fitted during training
        user_data[:, [0, 3, 4, 5, 8, 9]] = scaler.transform(
            user_data[:, [0, 3, 4, 5, 8, 9]]
        )

        # Get predictions from the trained models
        svm_model = models['svm']
        log_reg_model = models['log_reg']
        knn_model = models['knn']
        nn_model = models['nn']

        svm_prob = svm_model.predict_proba(user_data)[0][1]
        log_reg_prob = log_reg_model.predict_proba(user_data)[0][1]
        knn_prob = knn_model.predict_proba(user_data)[0][1]
        nn_prob = nn_model.predict(user_data)[0][0]

        threshold = 0.4  # You can adjust this threshold

        svm_pred = "High Risk" if svm_prob > threshold else "Low Risk"
        log_reg_pred = "High Risk" if log_reg_prob > threshold else "Low Risk"
        knn_pred = "High Risk" if knn_prob > threshold else "Low Risk"
        nn_pred = "High Risk" if nn_prob > threshold else "Low Risk"

        # Display results
        st.write("### Model Probabilities:")
        st.write(f"SVM Probability: {svm_prob:.2f}")
        st.write(f"Logistic Regression Probability: {log_reg_prob:.2f}")
        st.write(f"KNN Probability: {knn_prob:.2f}")
        st.write(f"Neural Network Probability: {nn_prob:.2f}")

        st.write("### Predictions:")
        st.write(f"SVM Model Prediction: {svm_pred}")
        st.write(f"Logistic Regression Prediction: {log_reg_pred}")
        st.write(f"KNN Model Prediction: {knn_pred}")
        st.write(f"Neural Network Prediction: {nn_pred}")
