import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

class OverviewAnalyser:   
         
    def missing_value(self,data):
        missing_value=data.isnull().sum()
        return missing_value
    
    def replace_outliers_with_mean(self,data, z_threshold=3):
        # Iterate through each numeric column
        for col in data.select_dtypes(include=[np.number]).columns:
            if col not in ['Bearer Id', 'IMSI', 'MSISDN/Number','IMEI']:
                col_data = data[col].dropna()
                col_zscore = zscore(col_data)
                
                # Create a boolean mask for outliers
                outlier_mask = abs(col_zscore) > z_threshold
                
                # Calculate the mean of non-outlier values (excluding NaNs)
                mean_value = col_data[~outlier_mask].mean()
                
                # Replace outliers in the original DataFrame
                # Need to align the original index with the calculated z-scores
                data.loc[data[col].notna() & (abs(zscore(data[col].fillna(0))) > z_threshold), col] = mean_value

        return data
    
    def replace_missing_with_mean_or_mode(self,data):
        #iterate over the columns
        for col in data.columns:
            if col in ['Bearer Id','IMSI']:
                data = data.dropna(subset=[col]) #drop the rows which don't have 'Bearer Id'or'IMSI'
            elif col in ['MSISDN/Number','IMEI','Last Location Name']:
                data[col] = data[col].fillna('Unknown') #replace with placeholder
            elif data[col].dtype == 'float64':  # If the column is numeric (float)
                mean_value = data[col].mean()
                data[col]=data[col].fillna(mean_value)  # Replace NaN with mean
            elif data[col].dtype == 'object':  # If the column is object (string)
                mode_value = data[col].mode()[0]
                data[col]=data[col].fillna(mode_value) # Replace NaN with mode
        
        return data
    
    def rank(self,data,column,rank):
        count=data[column].value_counts()
        return count.head(rank)
        
    def rank_bar_plot(self,data,column,title,x_label,y_label,rank=0):
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
        plt.show()
        

    def top_handsets_by_manufacturer(self,xdr_data):
        # Group by 'handset manufacturer' and 'handset type', and count occurrences
        handset_counts = xdr_data.groupby(['Handset Manufacturer', 'Handset Type']).size().reset_index(name='count')
        
        # Sort the counts by 'handset manufacturer' and 'count' in descending order
        handset_counts_sorted = handset_counts.sort_values(by=['Handset Manufacturer', 'count'],ascending=[True, False])
        
        # Get the top 3 manufacturers
        top_3_manufacturers = handset_counts_sorted['Handset Manufacturer'].value_counts().nlargest(3).index
        
        # Filter the data for the top 3 manufacturers
        top_3_handset_data = handset_counts_sorted[handset_counts_sorted['Handset Manufacturer'].isin(top_3_manufacturers)]
        
        # Get the top 5 handset types per manufacturer
        top_5_handsets_per_manufacturer = top_3_handset_data.groupby('Handset Manufacturer').head(5)
        
        return top_5_handsets_per_manufacturer

   

    def aggregate_xdr_data(self,data):
        # Grouping by 'IMSI' (assuming it identifies the user)
        user_aggregation = data.groupby('IMSI').agg(
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
    
    def segment_and_aggregate(self,data):
        # Ensure 'DL' and 'UL' columns are numeric
        data['Total DL (Bytes)'] = pd.to_numeric(data['Total DL (Bytes)'], errors='coerce')
        data['Total UL (Bytes)'] = pd.to_numeric(data['Total UL (Bytes)'], errors='coerce')
        
        # Compute total duration for all sessions and total data per user
        user_aggregates = data.groupby('IMSI').agg({
            'Dur. (ms)': 'sum',
            'Total DL (Bytes)': 'sum',
            'Total UL (Bytes)': 'sum'
        }).reset_index()
        
        # Compute total data (DL + UL) per user
        user_aggregates['Total Data (Bytes)'] = user_aggregates['Total DL (Bytes)'] + user_aggregates['Total UL (Bytes)']
        
        # Segment users into decile classes based on total duration
        user_aggregates['Decile'] = pd.qcut(user_aggregates['Dur. (ms)'], 10, labels=False) + 1
        
        # Compute the total data per decile class
        decile_data = user_aggregates.groupby('Decile').agg({
            'Total Data (Bytes)': 'sum'
        }).reset_index()
        
        return decile_data
    
    def decile_pie_chart(self,decile_data):
        # Plotting the pie chart
        fig, ax = plt.subplots()
        ax.pie(decile_data['Total Data (Bytes)'], 
            labels=decile_data['Decile'], 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=plt.cm.Paired(range(len(decile_data))))

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')

        plt.title('Distribution of Total Data by Decile')
        plt.show()
        
    def summary_statistics_excluding_columns(self,data, exclude_cols):
        # Select columns to include by excluding the specified columns
        include_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Calculate summary statistics
        summary = data[include_cols].describe()
        
        return summary

    def bivariate_analysis(self, aggregated_data):
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
            
            print("Correlation Matrix:")
            print(correlation_matrix['total_data'])
            
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
            plt.show()

    def correlation_matrix(self, aggregated_data):
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
        plt.show()
        
        return correlation_matrix
    
    def perform_pca(self, aggregated_data):
        # Combine download and upload data for each application
        aggregated_data['social_media_total'] = aggregated_data['social_media_dl'] + aggregated_data['social_media_ul']
        aggregated_data['google_total'] = aggregated_data['google_dl'] + aggregated_data['google_ul']
        aggregated_data['email_total'] = aggregated_data['email_dl'] + aggregated_data['email_ul']
        aggregated_data['youtube_total'] = aggregated_data['youtube_dl'] + aggregated_data['youtube_ul']
        aggregated_data['netflix_total'] = aggregated_data['netflix_dl'] + aggregated_data['netflix_ul']
        aggregated_data['gaming_total'] = aggregated_data['gaming_dl'] + aggregated_data['gaming_ul']
        aggregated_data['other_total'] = aggregated_data['other_dl'] + aggregated_data['other_ul']
        
        # Select relevant columns for PCA
        data_subset = aggregated_data[[
            'social_media_total', 'google_total', 'email_total', 
            'youtube_total', 'netflix_total', 'gaming_total', 'other_total'
        ]]
        
        # Step 1: Standardize the data
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(data_subset)
        
        # Step 2: Perform PCA
        pca = PCA(n_components=2)  # We'll reduce it to 2 components for simplicity
        principal_components = pca.fit_transform(standardized_data)
        
        # Step 3: Create a DataFrame for the principal components
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        
        # Plot the explained variance ratio
        plt.figure(figsize=(8, 6))
        plt.bar(range(1, 3), pca.explained_variance_ratio_, alpha=0.5, align='center', label='Individual explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.title('Explained Variance by Principal Components')
        plt.show()
        
        # Return PCA results and explained variance
        return pca_df, pca.explained_variance_ratio_

