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
import pandas as pd
from sqlalchemy import create_engine
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
    
    # return correlation_matrix

# useroverview_pair={
#         'Top 10 handsets used by the customers':20,
#         'Top 3 handset manufacturers':30,
#         'Bivariant Analysis':40,
#         'Correlation Analysis':50
#     }


#Sreamlit App
st.title("Telleco Data Analysis")

# Sidebar with options
st.sidebar.title("Analysis Options")
selected_analysis = st.sidebar.radio("Select Analysis", ["User Overview", "User Engagement", "User Expierience","User Satisfaction"])
xdr_data=load_table_to_dataframe('xdr_data_cleaned')

if selected_analysis == "User Overview":
        # xdr_data=load_table_to_dataframe('xdr_data_cleaned')
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
        st.sidebar.subheader("User Overview")
        selected_subanalysis = st.selectbox("Select a Subanalysis", 
                                            ['Top 10 handsets used by the customers',
                                            'Top 3 handset manufacturers',
                                            'Bivariate Analysis',
                                            'Correlation Analysis'])