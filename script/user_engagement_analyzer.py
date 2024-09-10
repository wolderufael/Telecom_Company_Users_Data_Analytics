import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class EngagementAnalyzer:
    def aggregate_engagement_metrics(self, data):
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

    def classify_customers_kmeans(self, data):
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
    
    def plot_clusters(self,customer_data):
        plt.scatter(customer_data['session_frequency'], customer_data['total_traffic'], 
                    c=customer_data['engagement_cluster'], cmap='viridis')
        plt.xlabel('Session Frequency')
        plt.ylabel('Total Traffic')
        plt.title('Customer Engagement Clusters')
        plt.colorbar(label='Cluster')
        plt.show()

    def cluster_summary_stats(self, classified_data):
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


    def plot_cluster_summary(self,cluster_stats):
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
            plt.show()

    def top_10_users_per_application(self, data):
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

    def plot_top_3_applications(self,top_10_users_data):
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
        plt.show()

    # Example usage:
    # plot_top_3_applications(top_10_users)


    def find_optimal_k(self,data, engagement_metrics):
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
        plt.show()

        return k_values, wcss

 

    def kmeans_clustering(self,data, engagement_metrics, optimal_k):
        metrics =engagement_metrics[['session_frequency','total_duration','total_traffic']]
        # Step 1: Normalize the engagement metrics
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(metrics)

        # Step 2: Run K-Means clustering with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        clusters = kmeans.fit_predict(normalized_data)

        # Step 3: Add the cluster labels to the original DataFrame
        engagement_metrics['Cluster'] = clusters

        return engagement_metrics

