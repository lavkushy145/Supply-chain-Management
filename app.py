import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.cluster import KMeans
# Suppress deprecation warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Supply Chain Management Dashboard", layout="wide")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", [
    "Homepage", "Data Upload", "Data Cleaning", "Inventory Optimization", "Customer Segmentation and Targeted Marketing"
])

def apply_label_encoders(data):
    encoders = {}
    for column in data.columns:
        if data[column].dtype == 'object':  # Assuming only categorical columns need encoding
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column].fillna('Missing'))  # Fill NA values
            encoders[column] = le
    return data, encoders

if page == "Homepage":
    st.title("Welcome to the Supply Chain Management Dashboard")
    if page == "Homepage":
    st.title("Welcome to the Supply Chain Management Dashboard")
    st.markdown("""
    ### Comprehensive Dashboard for Supply Chain Management

    This dashboard is designed to empower supply chain managers and teams with tools and insights necessary for effective supply chain management.

    **Navigate through the sidebar to access various functionalities:**

    - **Data Upload**: Upload your CSV data files here. Visualize and verify the data to ensure accuracy before processing.
    - **Data Cleaning**: Clean your data by removing duplicates, handling missing values, and preparing datasets for analysis.
    - **Inventory Optimization**: Utilize predictive analytics to forecast demand, calculate optimal reorder points, and manage inventory levels efficiently.
    - **Customer Segmentation and Targeted Marketing**: Segment your customers based on their behavior and demographics. Develop targeted marketing strategies to enhance customer engagement.
    - **Advanced Analytics**: Explore trend analysis, cost analysis, and other advanced analytical features to gain deeper insights into your supply chain operations.

    **Features Designed for Operational Excellence:**

    - **Automated Recommendations**: Receive actionable insights and automated recommendations based on the latest data.
    - **Real-time Updates**: The dashboard updates dynamically as new data is uploaded or existing data is modified, ensuring that you always have access to the latest insights.
    - **Collaborative Tool**: Share insights and reports directly from the dashboard to facilitate decision-making and streamline communication among team members and stakeholders.
    - **Customizable Visualizations**: Engage with interactive and customizable visual dashboards that cater to specific managerial needs.

    Use this dashboard to streamline your operations, reduce costs, and improve overall efficiency in managing your supply chain.
    """)


elif page == "Data Upload":
    st.title("Upload Supply Chain Data")
    uploaded_file = st.file_uploader("Choose a file", type=['csv'])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())
        st.session_state['data'] = data  # Save to session state for use in other pages

elif page == "Data Cleaning":
    st.title("Data Cleaning")
    if 'data' in st.session_state:
        data = st.session_state['data']
        st.write("Original Data", data.head())

        # Apply label encoders and handle errors
        try:
            data, label_encoders = apply_label_encoders(data)
            st.write("Data after encoding:", data.head())
            # Optionally, save encoders to a file for future use
            with open('label_encoders.pkl', 'wb') as file:
                pickle.dump(label_encoders, file)
        except Exception as e:
            st.error(f"An error occurred during encoding: {e}")
        
        # Display and handle missing values
        st.write("Missing values per column:")
        st.write(data.isnull().sum())
        if st.button("Remove Missing Values"):
            cleaned_data = data.dropna().reset_index(drop=True)
            st.session_state['cleaned_data'] = cleaned_data
            st.write("Data after removing missing values:", cleaned_data.head())
elif page == "Inventory Optimization":
    st.title("Inventory and Stock Optimization")
    if 'cleaned_data' in st.session_state:
        data = st.session_state['cleaned_data']
        features = ['Price', 'Availability', 'Number of products sold']
        X = data[features]
        y = data['Number of products sold']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        feature_importances = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            results[name] = mse
            if hasattr(model, 'feature_importances_'):
                feature_importances[name] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                feature_importances[name] = model.coef_
        
        st.subheader("Model Performance Comparison:")
        for name, mse in results.items():
            st.write(f"{name}: MSE = {mse}")
        
        fig, ax = plt.subplots()
        names = list(results.keys())
        values = list(results.values())
        ax.bar(names, values, color=['blue', 'green', 'red'])
        ax.set_ylabel('Mean Squared Error (MSE)')
        ax.set_title('Model Performance Comparison (MSE)')
        st.pyplot(fig)
        
        best_model = models[min(results, key=results.get)]
        
        st.subheader("Suggestions for Optimized Stock Levels and Reorder Quantities")
        if st.checkbox("Show Predictions for Optimization"):
            st.write("Enter the details for prediction:")
            price = st.number_input("Price", value=float(data['Price'].mean()))
            availability = st.number_input("Availability", value=int(data['Availability'].mean()))
            number_sold = st.number_input("Number of Products Sold", value=int(data['Number of products sold'].mean()))
            
            # Assembling input features based on label encoding
            encoded_price = price
            encoded_availability = availability
            encoded_number_sold = number_sold
            
            prediction = best_model.predict([[encoded_price, encoded_availability, encoded_number_sold]])
            recommended_stock = prediction[0] * 1.1
            st.write(f"Recommended reorder quantity: {recommended_stock:.2f} units.")
    else:
        st.error("No cleaned data available. Please complete the data cleaning step first.")
elif page == "Customer Segmentation and Targeted Marketing":
    st.title("Customer Segmentation and Targeted Marketing")
    if 'cleaned_data' in st.session_state:
        data = st.session_state['cleaned_data']

        # Assuming all required features have been encoded to numeric and we have their mappings
        features = ['Price', 'Availability', 'Number of products sold', 'Revenue generated', 'Lead times',
                    'Customer demographics', 'Product type', 'Location']
        X = data[features]

        if X.empty:
            st.error("No valid data available for modeling. Please check your data and selected features.")
        else:
            # Reverse mapping for displaying understandable results
            reverse_mapping = {
                'Customer demographics': {0: 'Non-binary', 1: 'Female', 2: 'Unknown', 3: 'Male'},
                'Product type': {0: 'Haircare', 1: 'Skincare', 2: 'Cosmetic'},
                'Location': {0: 'Mumbai', 1: 'Delhi', 2: 'Bangalore', 3: 'Kolkata', 4: 'Chennai'}
            }

            # Reverse map the encoded data for display
            for feature, mapping in reverse_mapping.items():
                if feature in data.columns:
                    data[feature] = data[feature].map(mapping)

            # Create pipelines for machine learning models
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))  # Scale relevant numeric features only

            # ANN model for predictions
            ann = Sequential([
                Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            ann.compile(optimizer='adam', loss='mean_squared_error')
            ann.fit(X_scaled, data['Number of products sold'], epochs=10, batch_size=10)

            # K-Means for clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            cluster_labels = kmeans.fit_predict(X_scaled)
            data['Cluster Labels'] = cluster_labels  # Append cluster labels to data


            # Display the clustered data with readable category names
            clustered_data = data[['Customer demographics', 'Product type', 'Location', 'Cluster Labels']]
            st.write("Segmented data:", clustered_data.head())

            # Analyze and display insights
            grouped_data = data.groupby(['Cluster Labels', 'Location', 'Product type']).agg({
                'Number of products sold': 'sum',
                'Revenue generated': 'mean'
            }).reset_index()

            st.subheader("Analysis of Product Sales and Revenue by Segment:")
            st.write(grouped_data)
            st.subheader("Data Visualization for Segmented Customer Data")

        # Cluster Distribution Plot
        fig, ax = plt.subplots()
        sns.countplot(x='Cluster Labels', data=data, palette='viridis', ax=ax)
        ax.set_title('Distribution of Data across Clusters')
        ax.set_xlabel('Cluster Label')
        ax.set_ylabel('Count')
        st.pyplot(fig)

        # Sales by Product Type and Location in Each Cluster
        fig, ax = plt.subplots()
        cluster_grouped = data.groupby(['Cluster Labels', 'Location', 'Product type']).sum().reset_index()
        sns.barplot(x='Location', y='Number of products sold', hue='Product type', data=cluster_grouped, ax=ax)
        ax.set_title('Sales by Product Type and Location per Cluster')
        ax.set_xlabel('Location')
        ax.set_ylabel('Total Products Sold')
        st.pyplot(fig)

        # Average Revenue Generated by Each Cluster
        fig, ax = plt.subplots()
        sns.barplot(x='Cluster Labels', y='Revenue generated', data=cluster_grouped, estimator=np.mean, ci=None, ax=ax)
        ax.set_title('Average Revenue Generated by Each Cluster')
        ax.set_xlabel('Cluster Label')
        ax.set_ylabel('Average Revenue')
        st.pyplot(fig)

        # Detailed insights visualization
        g = sns.FacetGrid(cluster_grouped, col='Cluster Labels', height=5, aspect=1)
        g.map_dataframe(sns.barplot, x='Location', y='Number of products sold', hue='Product type')
        g.add_legend()
        g.set_titles("Cluster {col_name}")
        g.set_axis_labels("Location", "Number of Products Sold")
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                    label.set_rotation(45)
        st.pyplot(g)

    else:
        st.error("No cleaned data available. Please upload and clean data first.")
                    # Additional Visualization for Clustered Data
        
