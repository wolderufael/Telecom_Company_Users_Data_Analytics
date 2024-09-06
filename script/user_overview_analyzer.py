import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class OverviewAnalyser:        
    def missing_value(self,data):
        missing_value=data.isnull().sum()
        return missing_value
    
    def replace_with_mean_or_mode(self,data):
        #iterate over the columns
        for col in data.columns:
            if col=='Bearer Id'or col=='IMSI':
                data = data.dropna(subset=[col]) #drop the rows which don't have 'Bearer Id'or'IMSI'
            elif col=='MSISDN/Number' or col=='IMEI' or col=='Last Location Name':
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


