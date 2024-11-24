import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv("dashboard/testdata.csv")

st.set_page_config(
    page_title='Medical dashboard',
    page_icon='🌟',
    layout='wide'
)
# TOP KPI's
total_patients = 500
discharged_count = df.loc[df['Status_Info'] == 'Discharged'].count()[0]
died_count = df.loc[df['Status_Info'] == 'Died'].count()[0]

# Calculate the percentage of positive reviews
yes_count = df['Positive_Review'].value_counts().get('Yes', 0)
total_reviews = df['Positive_Review'].count()  # Total count of reviews
positive_percentage = (yes_count / total_reviews) * 100  # Calculate percentage

# Calculating emergency Cases
Emergency_count = df.loc[df['Type_Of_Admission'] == 'Emergency'].count()[0]
st.markdown("<h1 style='margin-top: -10px;'>Dashboard 📊</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid #ccc;'>", unsafe_allow_html=True)
st.subheader('Metrics')
st.markdown("<hr style='border: 0.5px solid #ccc;'>", unsafe_allow_html=True)
# Create columns for layout

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Discharged 🛏️", value=discharged_count, delta=10, delta_color="normal")
    st.metric(label="No Of Patients Died ❌", value=died_count, delta=2, delta_color="inverse")

with col2:
    st.metric(label="Positve Reviews ✅", value=f"{positive_percentage:.2f}%", delta=1.5, delta_color="normal")
    st.metric(label="Revenue 💵", value=15324, delta=5000, delta_color="normal")

with col3:
    # st.metric(label="Avg Waiting Time ⌛", value=20, delta=-5.6, delta_color="normal")
    st.metric(label="Emergency Cases ⚠️", value=Emergency_count, delta=-8, delta_color="inverse")

st.markdown("<hr style='border: 2px solid #ccc;'>", unsafe_allow_html=True)
disease_filter = st.selectbox("select the disease", df["D_name"].unique())
placeholder = st.empty()

if disease_filter == 'Diabetes':
    filtered_df = df.query("D_name == 'Diabetes'")
elif disease_filter == 'cardio arrest':
    filtered_df = df.query("D_name == 'cardio arrest'")
elif disease_filter == 'respiratory problems':
    filtered_df = df.query("D_name == 'respiratory problems'")
elif disease_filter == 'Covid':
    filtered_df = df.query("D_name == 'Covid'")
elif disease_filter == 'Hypertension':
    filtered_df = df.query("D_name == 'Hypertension'")
elif disease_filter == 'TB':
    filtered_df = df.query("D_name == 'TB'")
elif disease_filter == 'Cancer':
    filtered_df = df.query("D_name == 'Cancer'")
elif disease_filter == 'Infant_and_Pregnancy cases':
    filtered_df = df.query("D_name == 'Infant_and_Pregnancy cases'")
elif disease_filter == 'accident_cases':
    filtered_df = df.query("D_name == 'accident_cases'")
else:
    filtered_df = df  # Handle the default case here

fig1, fig2, fig3 = st.columns(3)
with fig1:
    if filtered_df is not None:
        st.subheader('{}'.format(disease_filter))
        gender_counts = filtered_df['Gender'].value_counts()
        st.bar_chart(gender_counts)

with fig2:
    if filtered_df is not None:
        st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
        location_gender = pd.crosstab(filtered_df["Location"], df['Gender'])
        st.bar_chart(location_gender)
with fig3:
    if filtered_df is not None:
        #         st.subheader('Disease Distribution by Gender')
        st.markdown("<div style='height: 70px;'></div>", unsafe_allow_html=True)
        local_disease = pd.crosstab(filtered_df['Location'], filtered_df['D_name'])
        st.area_chart(local_disease)
st.markdown("<hr style='border: 2px solid #ccc;'>", unsafe_allow_html=True)
figu1, figu2 = st.columns(2)
with figu1:
    status_info_counts = filtered_df['Status_Info'].value_counts()
    fig_status_info_pie = px.pie(values=status_info_counts, names=status_info_counts.index)
    st.subheader('Status Info Distribution')
    st.plotly_chart(fig_status_info_pie)

with figu2:
    blood_gender = pd.crosstab(filtered_df['Blood_Type'], filtered_df['Gender'])
    fig_blood_gender = px.bar(blood_gender, barmode='group')
    st.subheader('Blood Type Distribution by Gender')
    st.plotly_chart(fig_blood_gender)

st.markdown("<hr style='border: 3px solid #ccc;'>", unsafe_allow_html=True)
# Select box to select visualization
visualization_option = st.selectbox('Select Visualization', ('Disease Distribution by Gender',
                                                             'Blood Distribution By Gender',
                                                             'Location Count',
                                                             'Addiction Count',
                                                             'Addiction vs. Gender',
                                                             'Addiction vs. Disease Type',
                                                             'Status Info Counts',
                                                             'Type of Admission Counts'))

if visualization_option == 'Disease Distribution by Gender':
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.subheader('Disease Distribution by Gender')
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    disease_gender = pd.crosstab(df['D_name'], df['Gender'])
    st.bar_chart(disease_gender)

elif visualization_option == 'Blood Distribution By Gender':
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.subheader('Blood Distribution By Gender')
    blood_gender = pd.crosstab(df['Blood_Type'], df['Gender'])
    st.bar_chart(blood_gender)

elif visualization_option == 'Location Count':
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.subheader('Count of patients by location')
    st.bar_chart(df['Location'].value_counts())

elif visualization_option == 'Addiction Count':
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.subheader('Count of patients by addiction status')
    st.bar_chart(df['Addiction'].value_counts())

elif visualization_option == 'Addiction vs. Gender':
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.subheader('Addiction Distribution by Gender')
    # Creating a cross-tab for addiction vs gender and visualizing it
    addiction_gender = pd.crosstab(df['Addiction'], df['Gender'])
    st.bar_chart(addiction_gender)

elif visualization_option == 'Addiction vs. Disease Type':
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.subheader('Addiction Distribution by Disease Type')
    # Creating a cross-tab for addiction vs gender and visualizing it
    addiction_disease = pd.crosstab(df['Addiction'], df['D_name'])
    st.bar_chart(addiction_disease)

elif visualization_option == 'Status Info Counts':
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.subheader('Status Info Counts')
    status_info_counts = df['Status_Info'].value_counts()
    st.bar_chart(status_info_counts)

elif visualization_option == 'Type of Admission Counts':
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.subheader('Type of Admission Counts')
    type_of_admission_counts = df['Type_Of_Admission'].value_counts()
    st.bar_chart(type_of_admission_counts)
st.markdown("<hr style='border: 2px solid #ccc;'>", unsafe_allow_html=True)

###

def process_and_visualize_data(df): #function that generates visuals based on user input data file
    st.title("Custom Hospital Patient Data Dashboard")

    # KPIs
    total_patients = len(df)
    status_counts = df['Status_Info'].value_counts()

    col1, col2 = st.columns(2)
    col1.metric("Total Patients", total_patients)
    for status, count in status_counts.items():
        col2.metric(f"{status} Count", count)

    # Gender Distribution
    st.header("Gender Distribution")
    gender_fig = px.pie(
        df,
        names='Gender',
        title='Gender Distribution'
    )
    st.plotly_chart(gender_fig)

    # Wait Time Analysis
    st.header("Wait Time Analysis")
    wait_time_fig = px.histogram(
        df,
        x='Wait_Time_Mins',
        title='Wait Time Distribution',
        nbins=10
    )
    st.plotly_chart(wait_time_fig)

    # Disease Analysis
    st.header("Disease Analysis")
    disease_data = df['D_name'].value_counts().reset_index()
    disease_data.columns = ['Disease', 'Patient Count']  # Rename columns for clarity
    disease_fig = px.bar(
        disease_data,
        x='Disease',
        y='Patient Count',
        title='Patients by Disease'
    )
    st.plotly_chart(disease_fig)

    # Preprocessing
    features = ['Age', 'DOS_Days', 'Wait_Time_Mins', 'Type_Of_Admission', 'Addiction', 'Gender', 'Blood_Type', 'Location', 'Positive_Review']
    target = 'Status_Info'
    # Drop rows with missing values in features or target
    df_clean = df.dropna(subset=features + [target])

    # Encoding for categorical features
    df_encoded = pd.get_dummies(df_clean, columns=['Type_Of_Admission', 'Addiction', 'Gender', 'Blood_Type', 'Location', 'Positive_Review'], drop_first=True)

    if 'D_name' in df_encoded.columns:
        df_encoded = pd.get_dummies(df_encoded, columns=['D_name'], drop_first=True)

    # Split data into train and test sets
    X = df_encoded.drop(columns=[target])
    y = df_clean[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Confusion Matrix Visualization
    st.header("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='black')
    ax.set_xticks(np.arange(len(model.classes_)))
    ax.set_yticks(np.arange(len(model.classes_)))
    ax.set_xticklabels(model.classes_)
    ax.set_yticklabels(model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

    # Classification Report
    st.header("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

    # Feature Importance
    st.header("Feature Importance")
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    fig_bar = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance'
    )
    st.plotly_chart(fig_bar)

# List of required columns for csv upload
expected_cols = [
    "D_name", "Gender", "Age", "Blood_Type", "Location",
    "Addiction", "Status_Info", "DOS_Days", "Type_Of_Admission",
    "Positive_Review", "Wait_Time_Mins"
]

# CSV Upload
st.sidebar.header("Insert Patient Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=None)

if uploaded_file:
    #check if file is a csv
    file_name = uploaded_file.name
    if not file_name.endswith(".csv"):
        st.sidebar.error("The uploaded file does not have the CSV extension. Please upload a valid CSV file.")
    else:
        try:
            user_data = pd.read_csv(uploaded_file)

            # make sure uploaded file has all expected columns
            missing_cols = []
            for col in expected_cols:
                if col not in user_data.columns:
                    missing_cols.append(col)
            if missing_cols:
                st.sidebar.error(
                    f"The uploaded file is missing the following required columns: {', '.join(missing_cols)}")
            else:
                st.sidebar.success("File uploaded successfully!")
                process_and_visualize_data(user_data)

        except Exception as e:
            st.sidebar.error(f"Error processing file: {e}")