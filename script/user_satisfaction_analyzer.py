from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pandas as pd

class SatisfactionAnalyzer:
    def user_aggregate(self,data):
        agg_data = data.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)':'mean',
            'TCP UL Retrans. Vol (Bytes)':'mean',
            'Avg Bearer TP DL (kbps)':'mean',
            'Avg Bearer TP UL (kbps)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Avg RTT UL (ms)': 'mean',
            'Handset Type': 'first'
        }).reset_index()
        
        #calculate average
        agg_data['Average TCP Retrans. Vol (Bytes)']=(agg_data['TCP DL Retrans. Vol (Bytes)']+agg_data['TCP UL Retrans. Vol (Bytes)'])/2
        agg_data['Avg Bearer TP (kbps)']=(agg_data['Avg Bearer TP DL (kbps)']+agg_data['Avg Bearer TP UL (kbps)'])/2
        agg_data[ 'Avg RTT (ms)']=(agg_data[ 'Avg RTT DL (ms)']+agg_data['Avg RTT UL (ms)'])/2

        return agg_data
    
    def kmeans_clustering(self,agg_data, n_clusters=3):
        # Selecting relevant columns for clustering
        experience_metrics = agg_data[['Average TCP Retrans. Vol (Bytes)', 'Avg RTT (ms)', 'Avg Bearer TP (kbps)']]
        
        # Normalizing the data
        scaler = MinMaxScaler()
        normalized_metrics = scaler.fit_transform(experience_metrics)
        
        # Applying KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        agg_data['Cluster'] = kmeans.fit_predict(normalized_metrics)
        
    # def calculate_engagement_experience_scores(self,agg_data, cluster_column='Cluster', low_engaged_cluster=0, worst_experience_cluster=0):
    #     engagement_scores = []
    #     experience_scores = []
        
    #     # Calculate Euclidean distance for each user
    #     for i, row in agg_data.iterrows():
    #         engagement_score = euclidean(
    #             row[['Average TCP Retrans. Vol (Bytes)', 'Avg RTT (ms)', 'Avg Bearer TP (kbps)']], 
    #             agg_data[agg_data[cluster_column] == low_engaged_cluster].iloc[0][['Average TCP Retrans. Vol (Bytes)', 'Avg RTT (ms)', 'Avg Bearer TP (kbps)']]
    #         )
    #         experience_score = euclidean(
    #             row[['Average TCP Retrans. Vol (Bytes)', 'Avg RTT (ms)', 'Avg Bearer TP (kbps)']], 
    #             agg_data[agg_data[cluster_column] == worst_experience_cluster].iloc[0][['Average TCP Retrans. Vol (Bytes)', 'Avg RTT (ms)', 'Avg Bearer TP (kbps)']]
    #         )
    #         engagement_scores.append(engagement_score)
    #         experience_scores.append(experience_score)
        
    #     agg_data['Engagement Score'] = engagement_scores
    #     agg_data['Experience Score'] = experience_scores
        
    #     return agg_data
    

    def calculate_engagement_experience_scores(self,agg_data, cluster_column='Cluster', low_engaged_cluster=0, worst_experience_cluster=0):
        # Extract the features used for distance calculation
        features = ['Average TCP Retrans. Vol (Bytes)', 'Avg RTT (ms)', 'Avg Bearer TP (kbps)']
        
        # Find the centroid for the low engaged and worst experience clusters
        low_engaged_centroid = agg_data[agg_data[cluster_column] == low_engaged_cluster][features].mean().values
        worst_experience_centroid = agg_data[agg_data[cluster_column] == worst_experience_cluster][features].mean().values
       
        #for debugging
        print("Low-engaged centroid:", low_engaged_centroid)
        print("Worst-experience centroid:", worst_experience_centroid)
        
        # Calculate the Euclidean distance in a vectorized manner
        feature_matrix = agg_data[features].values
        engagement_scores = np.linalg.norm(feature_matrix - low_engaged_centroid, axis=1)
        experience_scores = np.linalg.norm(feature_matrix - worst_experience_centroid, axis=1)
        
        # Normalize the engagement and experience scores to the range (-1, 1)
        def normalize_scores(scores):
            min_score = scores.min()
            max_score = scores.max()
            return 2 * (scores - min_score) / (max_score - min_score) - 1
        
        # Assign the calculated scores to the DataFrame
        agg_data['Engagement Score'] = normalize_scores(engagement_scores)
        agg_data['Experience Score'] = normalize_scores(experience_scores)
        
        return agg_data

    
    def calculate_satisfaction_scores(self, agg_data):
        # Average of both engagement & experience scores
        agg_data['Satisfaction Score'] = (agg_data['Engagement Score'] + agg_data['Experience Score']) / 2
        
        # Top 10 satisfied customers
        top_satisfied_customers = agg_data.nlargest(10, 'Satisfaction Score').reset_index()
        
        # Plotting the top 10 satisfied customers
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Satisfaction Score', y=top_satisfied_customers.index, 
                    data=top_satisfied_customers, hue='Satisfaction Score', palette='Blues_d', legend=False)
        plt.title('Top 10 Satisfied Customers')
        plt.xlabel('Satisfaction Score')
        plt.ylabel('Customer Index')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def regression_model(self,agg_data):
        # Prepare the data
        X = agg_data[['Average TCP Retrans. Vol (Bytes)', 'Avg RTT (ms)', 'Avg Bearer TP (kbps)', 'Engagement Score', 'Experience Score']]
        y = agg_data['Satisfaction Score']

        # Fit regression model
        model = LinearRegression()
        model.fit(X, y)
        
        return model
    
    def kmeans_on_engagement_experience(self, agg_data, n_clusters=2):
        # Selecting relevant columns for clustering
        experience_metrics = agg_data[['Engagement Score', 'Experience Score']]
        
        # Normalizing the data
        scaler = MinMaxScaler()
        normalized_metrics = scaler.fit_transform(experience_metrics)
        
        # Applying KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        agg_data['Satisfaction Cluster'] = kmeans.fit_predict(normalized_metrics)
        
        # Plotting the clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Engagement Score', y='Experience Score', hue='Cluster', data=agg_data, palette='viridis', s=100)
        plt.title(f'KMeans Clustering with {n_clusters} Clusters')
        plt.xlabel('Engagement Score')
        plt.ylabel('Experience Score')
        plt.legend(title='Cluster')
        plt.tight_layout()
        plt.show()

    def aggregate_cluster_scores(self, agg_data, cluster_column='Satisfaction Cluster'):
        # Aggregate data
        cluster_scores = agg_data.groupby(cluster_column).agg({
            'Satisfaction Score': 'mean',
            'Experience Score': 'mean'
        }).reset_index()

        # Plot the aggregated scores
        plt.figure(figsize=(10, 6))
        sns.barplot(x=cluster_column, y='Satisfaction Score', data=cluster_scores, color='skyblue', label='Satisfaction Score')
        sns.barplot(x=cluster_column, y='Experience Score', data=cluster_scores, color='salmon', label='Experience Score')

        plt.title('Average Satisfaction and Experience Scores by Cluster')
        plt.ylabel('Score')
        plt.legend()
        plt.show()

        # return cluster_scores
    
    def export_to_postgres(self,agg_data):
        # Create connection to PostgreSQL using SQLAlchemy
        engine = create_engine('postgresql+psycopg2://username:password@localhost:5432/database_name')
        connection = engine.connect()

        # Optional: Create a session (if needed)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Convert pandas DataFrame to SQL format and insert it into the PostgreSQL table
        agg_data[['MSISDN/Number', 'Engagement Score', 'Experience Score', 'Satisfaction Score']].to_sql(
            'user_scores', con=engine, index=False, if_exists='append', method='multi'
        )

        # Close connection and session
        session.close()
        connection.close()
