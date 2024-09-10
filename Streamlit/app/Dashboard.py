import os
os.environ['LOKY_MAX_CPU_COUNT'] = '2'
# notebook_dir = os.getcwd()
# parent_path=os.path.dirname(notebook_dir)
# os.chdir(parent_path)

import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
# Load environment variables from .env file
load_dotenv()

# Fetch credentials from .env file
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')

def load_table_to_dataframe(table_name):
    try:
        # Create an SQLAlchemy engine
        engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

        # Use pandas read_sql to load the table into a DataFrame
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql(query, engine)

        return df

    except Exception as error:
        print("Error while connecting to PostgreSQL", error)

    finally:
        # Close the connection
        engine.dispose()
        print("SQLAlchemy connection is disposed")
 #########################################################
 #overview       
def rank_bar_plot(data,column,title,x_label,y_label,rank=0):
    if rank==0:
        rank=len(data)  #if rank is not given take the whole length of data as rank
    else:
        rank=rank
    count = data[column].value_counts()
    plt.figure(figsize=(10, 6))
    # count.head(rank).plot(kind='bar', color=plt.cm.viridis(range(len(count.head()))))
    ax = count.head(rank).plot(kind='bar', color=plt.cm.viridis(range(len(count.head()))))

    # Annotate each bar with its height
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', va='baseline', xytext=(0, 5), textcoords='offset points')
    plt.title(f'{title}')
    plt.xlabel(f'{x_label}')
    plt.ylabel(f'{y_label}')
    st.pyplot(plt.gcf())

def aggregate_xdr_data(data):
    # Grouping by 'IMSI' (assuming it identifies the user)
    user_aggregation = data.groupby('MSISDN/Number').agg(
        num_sessions=('Bearer Id', 'count'),  # Number of xDR sessions
        total_duration=('Dur. (ms)', 'sum'),  # Total session duration
        total_dl_data=('Total DL (Bytes)', 'sum'),  # Total download data
        total_ul_data=('Total UL (Bytes)', 'sum'),  # Total upload data
        
        # Total data volume per application
        social_media_dl=('Social Media DL (Bytes)', 'sum'),
        social_media_ul=('Social Media UL (Bytes)', 'sum'),
        google_dl=('Google DL (Bytes)', 'sum'),
        google_ul=('Google UL (Bytes)', 'sum'),
        email_dl=('Email DL (Bytes)', 'sum'),
        email_ul=('Email UL (Bytes)', 'sum'),
        youtube_dl=('Youtube DL (Bytes)', 'sum'),
        youtube_ul=('Youtube UL (Bytes)', 'sum'),
        netflix_dl=('Netflix DL (Bytes)', 'sum'),
        netflix_ul=('Netflix UL (Bytes)', 'sum'),
        gaming_dl=('Gaming DL (Bytes)', 'sum'),
        gaming_ul=('Gaming UL (Bytes)', 'sum'),
        other_dl=('Other DL (Bytes)', 'sum'),
        other_ul=('Other UL (Bytes)', 'sum')
    ).reset_index()
    
    return user_aggregation
    
def bivariate_analysis(aggregated_data):
    #  Create Total Data Usage Columns (Total DL + UL)
    aggregated_data['total_data'] = aggregated_data['total_dl_data'] + aggregated_data['total_ul_data']
    
    # Creating total columns for each application
    aggregated_data['social_media_total'] = aggregated_data['social_media_dl'] + aggregated_data['social_media_ul']
    aggregated_data['google_total'] = aggregated_data['google_dl'] + aggregated_data['google_ul']
    aggregated_data['email_total'] = aggregated_data['email_dl'] + aggregated_data['email_ul']
    aggregated_data['youtube_total'] = aggregated_data['youtube_dl'] + aggregated_data['youtube_ul']
    aggregated_data['netflix_total'] = aggregated_data['netflix_dl'] + aggregated_data['netflix_ul']
    aggregated_data['gaming_total'] = aggregated_data['gaming_dl'] + aggregated_data['gaming_ul']
    aggregated_data['other_total'] = aggregated_data['other_dl'] + aggregated_data['other_ul']
    
    #  Correlation Analysis
    application_columns = [
        'social_media_total', 'google_total', 'email_total', 
        'youtube_total', 'netflix_total', 'gaming_total', 'other_total'
    ]
    
    # Correlation matrix
    correlation_matrix = aggregated_data[['total_data'] + application_columns].corr()
    
    #  Scatter Plots for Bivariate Analysis
    fig, axs = plt.subplots(4, 2, figsize=(15, 15))
    fig.suptitle('Bivariate Analysis: Total Data vs Application Data', fontsize=16)
    
    app_titles = ['Social Media', 'Google', 'Email', 'YouTube', 'Netflix', 'Gaming', 'Other']
    
    for i, app in enumerate(application_columns):
        row, col = divmod(i, 2)
        axs[row, col].scatter(aggregated_data[app], aggregated_data['total_data'])
        axs[row, col].set_title(f'{app_titles[i]} vs Total Data')
        axs[row, col].set_xlabel(f'{app_titles[i]} Data')
        axs[row, col].set_ylabel('Total Data')

    # Remove empty subplot
    fig.delaxes(axs[3][1])
    
    plt.tight_layout()
    st.pyplot(plt.gcf())

def correlation_matrix(aggregated_data):
    # Combine download and upload data to get total data for each application
    aggregated_data['social_media_total'] = aggregated_data['social_media_dl'] + aggregated_data['social_media_ul']
    aggregated_data['google_total'] = aggregated_data['google_dl'] + aggregated_data['google_ul']
    aggregated_data['email_total'] = aggregated_data['email_dl'] + aggregated_data['email_ul']
    aggregated_data['youtube_total'] = aggregated_data['youtube_dl'] + aggregated_data['youtube_ul']
    aggregated_data['netflix_total'] = aggregated_data['netflix_dl'] + aggregated_data['netflix_ul']
    aggregated_data['gaming_total'] = aggregated_data['gaming_dl'] + aggregated_data['gaming_ul']
    aggregated_data['other_total'] = aggregated_data['other_dl'] + aggregated_data['other_ul']
    
    # Select only the total data columns
    data_subset = aggregated_data[[
        'social_media_total', 'google_total', 'email_total', 
        'youtube_total', 'netflix_total', 'gaming_total', 'other_total'
    ]]
    
    # Compute the correlation matrix
    correlation_matrix = data_subset.corr()
    
    # Visualize the correlation matrix using a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Application Data Usage')
    st.pyplot(plt.gcf())
#overview
##############################################################
    
###############################################################
#engagement
def aggregate_engagement_metrics(data):
    # Step 1: Ensure 'Total DL (Bytes)' and 'Total UL (Bytes)' are numeric
    data['Total DL (Bytes)'] = pd.to_numeric(data['Total DL (Bytes)'], errors='coerce')
    data['Total UL (Bytes)'] = pd.to_numeric(data['Total UL (Bytes)'], errors='coerce')
    
    # Step 2: Aggregate session metrics per customer (MSISDN)
    customer_aggregation = data.groupby('MSISDN/Number').agg(
        session_frequency=('Bearer Id', 'count'),  # Number of sessions
        total_duration=('Dur. (ms)', 'sum'),  # Total session duration (in milliseconds)
        total_dl=('Total DL (Bytes)', 'sum'),  # Total download traffic
        total_ul=('Total UL (Bytes)', 'sum')   # Total upload traffic
    ).reset_index()
    
    # Step 3: Calculate total traffic (download + upload) after aggregation
    customer_aggregation['total_traffic'] = customer_aggregation['total_dl'] + customer_aggregation['total_ul']
    
    # Step 4: Sort and report top 10 customers for each metric
    
    # Top 10 customers by session frequency
    top_10_session_frequency = customer_aggregation.nlargest(10, 'session_frequency')[['MSISDN/Number', 'session_frequency']]
    
    # Top 10 customers by total session duration
    top_10_session_duration = customer_aggregation.nlargest(10, 'total_duration')[['MSISDN/Number', 'total_duration']]
    
    # Top 10 customers by total traffic (download + upload)
    top_10_total_traffic = customer_aggregation.nlargest(10, 'total_traffic')[['MSISDN/Number', 'total_traffic']]
    
    return customer_aggregation,top_10_session_frequency, top_10_session_duration, top_10_total_traffic
def classify_customers_kmeans(data):
    # Step 1: Aggregate the engagement metrics (from the previous function)
    customer_aggregation = data.groupby('MSISDN/Number').agg(
        session_frequency=('Bearer Id', 'count'),  # Number of sessions
        total_duration=('Dur. (ms)', 'sum'),  # Total session duration (in milliseconds)
        total_dl=('Total DL (Bytes)', 'sum'),  # Total download traffic
        total_ul=('Total UL (Bytes)', 'sum')   # Total upload traffic
    ).reset_index()
    
    # Calculate total traffic
    customer_aggregation['total_traffic'] = customer_aggregation['total_dl'] + customer_aggregation['total_ul']
    
    # Select only the engagement metrics
    engagement_metrics = customer_aggregation[['session_frequency', 'total_duration', 'total_traffic']]
    
    # Step 2: Normalize the engagement metrics
    scaler = MinMaxScaler()
    normalized_metrics = scaler.fit_transform(engagement_metrics)
    
    # Step 3: Apply K-Means clustering with k=3
    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_aggregation['engagement_cluster'] = kmeans.fit_predict(normalized_metrics)
    
    # Step 4: Return the DataFrame with cluster assignments
    return customer_aggregation[['MSISDN/Number', 'session_frequency', 'total_duration', 'total_traffic', 'engagement_cluster']]

def plot_clusters(customer_data):
    plt.scatter(customer_data['session_frequency'], customer_data['total_traffic'], 
                c=customer_data['engagement_cluster'], cmap='viridis')
    plt.xlabel('Session Frequency')
    plt.ylabel('Total Traffic')
    plt.title('Customer Engagement Clusters')
    plt.colorbar(label='Cluster')
    st.pyplot(plt.gcf())
    
def cluster_summary_stats(classified_data):
# Step 1: Compute summary statistics for each cluster
    cluster_stats = classified_data.groupby('engagement_cluster').agg(
    min_session_frequency=('session_frequency', 'min'),
    max_session_frequency=('session_frequency', 'max'),
    avg_session_frequency=('session_frequency', 'mean'),
    total_session_frequency=('session_frequency', 'sum'),
    
    min_total_duration=('total_duration', 'min'),
    max_total_duration=('total_duration', 'max'),
    avg_total_duration=('total_duration', 'mean'),
    total_total_duration=('total_duration', 'sum'),
    
    min_total_traffic=('total_traffic', 'min'),
    max_total_traffic=('total_traffic', 'max'),
    avg_total_traffic=('total_traffic', 'mean'),
    total_total_traffic=('total_traffic', 'sum')
        ).reset_index()

    return cluster_stats


def plot_cluster_summary(cluster_stats):
    metrics = ['session_frequency', 'total_duration', 'total_traffic']
    
    for metric in metrics:
        # Create subplots for each metric
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot Min, Max, and Avg for each cluster
        cluster_stats.plot(kind='bar', x='engagement_cluster', y=[f'min_{metric}', f'max_{metric}', f'avg_{metric}'], ax=ax[0])
        ax[0].set_title(f'Min, Max, and Avg {metric.capitalize()} per Cluster')
        ax[0].set_ylabel(f'{metric.capitalize()}')
        
        # Plot Total for each cluster
        cluster_stats.plot(kind='bar', x='engagement_cluster', y=f'total_{metric}', ax=ax[1], color='orange')
        ax[1].set_title(f'Total {metric.capitalize()} per Cluster')
        ax[1].set_ylabel(f'Total {metric.capitalize()}')
        
        plt.tight_layout()
        st.pyplot(plt.gcf())
        
 
def top_10_users_per_application(data):
    # Step 1: Ensure all relevant columns are numeric
    columns_to_convert = [
        'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
        'Google DL (Bytes)', 'Google UL (Bytes)',
        'Email DL (Bytes)', 'Email UL (Bytes)',
        'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
        'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
        'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
        'Other DL (Bytes)', 'Other UL (Bytes)'
    ]
    
    # Convert to numeric and handle errors
    for col in columns_to_convert:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Step 2: Aggregate download and upload separately
    user_application_traffic = data.groupby('MSISDN/Number').agg(
        social_media_dl=('Social Media DL (Bytes)', 'sum'),
        social_media_ul=('Social Media UL (Bytes)', 'sum'),
        google_dl=('Google DL (Bytes)', 'sum'),
        google_ul=('Google UL (Bytes)', 'sum'),
        email_dl=('Email DL (Bytes)', 'sum'),
        email_ul=('Email UL (Bytes)', 'sum'),
        youtube_dl=('Youtube DL (Bytes)', 'sum'),
        youtube_ul=('Youtube UL (Bytes)', 'sum'),
        netflix_dl=('Netflix DL (Bytes)', 'sum'),
        netflix_ul=('Netflix UL (Bytes)', 'sum'),
        gaming_dl=('Gaming DL (Bytes)', 'sum'),
        gaming_ul=('Gaming UL (Bytes)', 'sum'),
        other_dl=('Other DL (Bytes)', 'sum'),
        other_ul=('Other UL (Bytes)', 'sum')
    ).reset_index()
    
    # Step 3: Calculate total traffic (download + upload) for each application
    user_application_traffic['total_social_media_traffic'] = user_application_traffic['social_media_dl'] + user_application_traffic['social_media_ul']
    user_application_traffic['total_google_traffic'] = user_application_traffic['google_dl'] + user_application_traffic['google_ul']
    user_application_traffic['total_email_traffic'] = user_application_traffic['email_dl'] + user_application_traffic['email_ul']
    user_application_traffic['total_youtube_traffic'] = user_application_traffic['youtube_dl'] + user_application_traffic['youtube_ul']
    user_application_traffic['total_netflix_traffic'] = user_application_traffic['netflix_dl'] + user_application_traffic['netflix_ul']
    user_application_traffic['total_gaming_traffic'] = user_application_traffic['gaming_dl'] + user_application_traffic['gaming_ul']
    user_application_traffic['total_other_traffic'] = user_application_traffic['other_dl'] + user_application_traffic['other_ul']
    
    # Step 4: Find the top 10 users for each application
    top_10_social_media = user_application_traffic.nlargest(10, 'total_social_media_traffic')[['MSISDN/Number', 'total_social_media_traffic']]
    top_10_google = user_application_traffic.nlargest(10, 'total_google_traffic')[['MSISDN/Number', 'total_google_traffic']]
    top_10_email = user_application_traffic.nlargest(10, 'total_email_traffic')[['MSISDN/Number', 'total_email_traffic']]
    top_10_youtube = user_application_traffic.nlargest(10, 'total_youtube_traffic')[['MSISDN/Number', 'total_youtube_traffic']]
    top_10_netflix = user_application_traffic.nlargest(10, 'total_netflix_traffic')[['MSISDN/Number', 'total_netflix_traffic']]
    top_10_gaming = user_application_traffic.nlargest(10, 'total_gaming_traffic')[['MSISDN/Number', 'total_gaming_traffic']]
    top_10_other = user_application_traffic.nlargest(10, 'total_other_traffic')[['MSISDN/Number', 'total_other_traffic']]
    
    # Step 5: Return the top 10 users for each application
    return {
        'Social Media': top_10_social_media,
        'Google': top_10_google,
        'Email': top_10_email,
        'YouTube': top_10_youtube,
        'Netflix': top_10_netflix,
        'Gaming': top_10_gaming,
        'Other': top_10_other
    }

def plot_top_3_applications(top_10_users_data):
    # Step 1: Calculate total traffic for each application
    app_traffic = {
        'Social Media': top_10_users_data['Social Media']['total_social_media_traffic'].sum(),
        'Google': top_10_users_data['Google']['total_google_traffic'].sum(),
        'Email': top_10_users_data['Email']['total_email_traffic'].sum(),
        'YouTube': top_10_users_data['YouTube']['total_youtube_traffic'].sum(),
        'Netflix': top_10_users_data['Netflix']['total_netflix_traffic'].sum(),
        'Gaming': top_10_users_data['Gaming']['total_gaming_traffic'].sum(),
        'Other': top_10_users_data['Other']['total_other_traffic'].sum()
    }

    # Step 2: Sort applications by total traffic and pick the top 3
    top_3_apps = sorted(app_traffic.items(), key=lambda x: x[1], reverse=True)[:3]

    # Step 3: Create a DataFrame for easy plotting
    top_3_apps_df = pd.DataFrame(top_3_apps, columns=['Application', 'Total Traffic (Bytes)'])

    # Step 4: Plot a bar chart for the top 3 applications
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Application', y='Total Traffic (Bytes)', data=top_3_apps_df)

    plt.title('Top 3 Most Used Applications by Total Traffic', fontsize=16)
    plt.ylabel('Total Traffic (Bytes)', fontsize=12)
    plt.xlabel('Application', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    
def find_optimal_k(data, engagement_metrics):
    metrics =engagement_metrics[['session_frequency','total_duration','total_traffic']]
    # Step 1: Normalize the engagement metrics (features)
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(metrics)

    # Step 2: Use KMeans to cluster the data and calculate WCSS for different values of k
    wcss = []
    k_values = range(1, 11)  # You can try more k values if needed
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(normalized_data)
        wcss.append(kmeans.inertia_)  # Inertia is the WCSS

    # Step 3: Plot the Elbow graph
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal K', fontsize=16)
    plt.xlabel('Number of clusters (k)', fontsize=12)
    plt.ylabel('WCSS', fontsize=12)
    plt.xticks(k_values)
    plt.grid(True)
    st.pyplot(plt.gcf())

    # return k_values, wcss
#engagement
################################################################

#Sreamlit App
st.title("Telleco Data Analysis")

# Sidebar with options
st.sidebar.title("Analysis Options")
selected_analysis = st.sidebar.radio("Select Analysis", ["User Overview", "User Engagement", "User Expierience","User Satisfaction"])
xdr_data=load_table_to_dataframe('xdr_data_cleaned')

if selected_analysis == "User Overview":
        aggregated_xdr_data= aggregate_xdr_data(xdr_data)
 
        st.sidebar.subheader("User Overview")
        selected_subanalysis = st.selectbox("Select a Subanalysis", 
                                            ['Top 10 handsets used by the customers',
                                            'Top 3 handset manufacturers',
                                            'Bivariate Analysis',
                                            'Correlation Analysis'])

        # Execute only the selected analysis
        if selected_subanalysis == 'Top 10 handsets used by the customers':
            rank_bar_plot(xdr_data, 'Handset Type', 'Top 10 handsets used by the customers', 'Handset Types', 'Number of Users', 10)
        elif selected_subanalysis == 'Top 3 handset manufacturers':
            rank_bar_plot(xdr_data, 'Handset Manufacturer', 'Top 3 Handset Manufacturers', 'Handset Manufacturer', 'Number of Users', 3)
        elif selected_subanalysis == 'Bivariate Analysis':
            bivariate_analysis(aggregated_xdr_data)
        elif selected_subanalysis == 'Correlation Analysis':
            correlation_matrix(aggregated_xdr_data)
            
if selected_analysis=="User Engagement":
        customer_aggregation,top_10_freq, top_10_duration, top_10_traffic = aggregate_engagement_metrics(xdr_data)
        classified_customers =classify_customers_kmeans(xdr_data)
        st.sidebar.subheader("User Engagement")
        selected_subanalysis = st.selectbox("Select a Subanalysis", 
                                            ['Customer Engagement Cluster',
                                            'Cluster Stats',
                                            'Elbow Method for Optimal K'])
        
        if selected_subanalysis=='Customer Engagement Cluster':
            plot_clusters(classified_customers)
        elif selected_subanalysis=='Cluster Stats':
            cluster_stats = cluster_summary_stats(classified_customers)
            plot_cluster_summary(cluster_stats)
        elif selected_subanalysis=='Top 3 Most Used Applications by Total Traffic':
            top_10_users = top_10_users_per_application(xdr_data)
            plot_top_3_applications(top_10_users)
        elif selected_subanalysis=='Elbow Method for Optimal K':
            engagement_metrics=customer_aggregation
            find_optimal_k(xdr_data, engagement_metrics)