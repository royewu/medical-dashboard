# A Medical Dashboard For Hospital Management
##### Enhancing Decision Making Of Hospitals By Providing Insights Through Usage of a Medical Dashboard

### Introduction
This Streamlit application is a medical dashboard for envisioning and analyzing hospital patient data. It provides various metrics, visualizations, and insights based on the dataset provided.

###  Features
- Metrics: Display key performance indicators (KPIs) such as discharged patient count, death count, positive reviews percentage, etc.
- Charts: Visualize data using different types of charts, including bar charts, pie charts, area charts, and more.
- Filtering: Select specific diseases and view corresponding visualizations.
- Disease Distribution: Analyze disease distribution by gender, blood type, and location.
- Addiction Analysis: Explore addiction counts and distribution by gender.
- Status Info Distribution: View status information distribution and counts.
- Type of Admission Analysis: Analyze counts of different types of admissions.
- Custom Patient Data Analysis Tool: Allows for user to upload patient data in the form of a csv file and will analyze data and provide the following:
    -  Important KPI's, including counts of total patients, ICU, admitted, discharged, and died
    -  Gender Distribution, Wait Time Analysis, and Disease Analysis charts
    -  Utilizes machine learning algorithm model (random forest) to generate the following
        -  Confusion Matrix: to evaluate the model's performance
        -  Classification Report: to summarize the model's performance
        -  Feature Importance: to determine which features have the highest impact on the data

### How to Run

##### Requirements:
Python 3.x
Required packages: see requirements.txt file

##### Running the Application:
Through command line use the following command:
streamlit run dashboard.py

##### Using the Application:
The application will open in your default web browser.
You can interact with the filters and visualizations to explore the hospital management data.
You can upload a csv with the same format as testdata.csv to analyze your own data (csv file called sample_data1.csv is there for testing within the data folder)
##### Data Source
The application uses a CSV file named testdata.csv to load the hospital management data (it is within the data folder). Please make sure this file is present in the data folder, or change the directory to match the csv file.

##### Customization
Customization of the visualizations, metrics, and data filters based on your specific requirements is possible through by modifying the code in dashboard.py.
