import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class ExperienceAnalyzer:
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

        print(agg_data.head())
        return agg_data
        
    def tcp_rtt_throughput_analysis(self,agg_data):
        # Top, Bottom, and Most Frequent TCP, RTT, and Throughput
        top_tcp = agg_data.nlargest(10, 'Average TCP Retrans. Vol (Bytes)')
        bottom_tcp = agg_data.nsmallest(10, 'Average TCP Retrans. Vol (Bytes)')
        freq_tcp = agg_data['Average TCP Retrans. Vol (Bytes)'].value_counts().head(10)
        
        top_rtt = agg_data.nlargest(10, 'Avg RTT (ms)')
        bottom_rtt = agg_data.nsmallest(10, 'Avg RTT (ms)')
        freq_rtt = agg_data['Avg RTT (ms)'].value_counts().head(10)

        top_throughput = agg_data.nlargest(10, 'Avg Bearer TP (kbps)')
        bottom_throughput = agg_data.nsmallest(10, 'Avg Bearer TP (kbps)')
        freq_throughput = agg_data['Avg Bearer TP (kbps)'].value_counts().head(10)

        return {
            'Top TCP': top_tcp, 
            'Bottom TCP': bottom_tcp, 
            'Most Frequent TCP': freq_tcp,
            'Top RTT': top_rtt, 
            'Bottom RTT': bottom_rtt, 
            'Most Frequent RTT': freq_rtt,
            'Top Throughput': top_throughput, 
            'Bottom Throughput': bottom_throughput, 
            'Most Frequent Throughput': freq_throughput
        }


    def throughput_tcp_handset_analysis(self,agg_data):
        # Distribution of average throughput per handset type (Top 10)
        throughput_dist = agg_data.groupby('Handset Type')['Avg Bearer TP (kbps)'].mean().reset_index()
        top10_throughput_dist = throughput_dist.nlargest(10, 'Avg Bearer TP (kbps)')

        # Average TCP retransmission per handset type (Top 10)
        tcp_retrans_dist = agg_data.groupby('Handset Type')['Average TCP Retrans. Vol (Bytes)'].mean().reset_index()
        top10_tcp_retrans_dist = tcp_retrans_dist.nlargest(10, 'Average TCP Retrans. Vol (Bytes)')

        # Plot the distribution of average throughput for top 10 handsets
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Avg Bearer TP (kbps)', y='Handset Type', hue='Handset Type', data=top10_throughput_dist, legend=False)
        plt.title('Top 10 Handsets by Average Bearer Throughput (kbps)')
        plt.xlabel('Average Bearer TP (kbps)')
        plt.ylabel('Handset Type')
        plt.tight_layout()
        plt.show()

        # Plot the distribution of average TCP retransmission for top 10 handsets
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Average TCP Retrans. Vol (Bytes)', y='Handset Type', hue='Handset Type', data=top10_tcp_retrans_dist, legend=False)
        plt.title('Top 10 Handsets by Average TCP Retransmission Volume (Bytes)')
        plt.xlabel('Average TCP Retrans. Vol (Bytes)')
        plt.ylabel('Handset Type')
        plt.tight_layout()
        plt.show()


    # def kmeans_clustering(self,agg_data, n_clusters=3):
    #     # Selecting relevant columns for clustering
    #     experience_metrics = agg_data[['Average TCP Retrans. Vol (Bytes)', 'Avg RTT (ms)', 'Avg Bearer TP (kbps)']]
        
    #     # Normalizing the data
    #     scaler = MinMaxScaler()
    #     normalized_metrics = scaler.fit_transform(experience_metrics)
        
    #     # Applying KMeans clustering
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    #     agg_data['Cluster'] = kmeans.fit_predict(normalized_metrics)
        
    #     # Plotting the clusters
    #     plt.figure(figsize=(12, 8))
        
    #     # Plot for TCP Retransmission vs. Avg RTT
    #     plt.subplot(1, 2, 1)
    #     sns.scatterplot(x='Average TCP Retrans. Vol (Bytes)', y='Avg RTT (ms)', hue='Cluster', data=agg_data, palette='viridis', s=100, alpha=0.7)
    #     plt.title('TCP Retransmission vs. Avg RTT')
        
    #     # Plot for Avg Bearer TP vs. Avg RTT
    #     plt.subplot(1, 2, 2)
    #     sns.scatterplot(x='Avg Bearer TP (kbps)', y='Avg RTT (ms)', hue='Cluster', data=agg_data, palette='viridis', s=100, alpha=0.7)
    #     plt.title('Avg Bearer TP vs. Avg RTT')
        
    #     plt.tight_layout()
    #     plt.show()
        
    #     # Describing each cluster
    #     cluster_summary = agg_data.groupby('Cluster')
    #     # .mean().reset_index()\
        
    #     # return agg_data, cluster_summary
    def kmeans_clustering(self, agg_data, n_clusters=3):
        # Selecting relevant columns for clustering
        experience_metrics = agg_data[['Average TCP Retrans. Vol (Bytes)', 'Avg RTT (ms)', 'Avg Bearer TP (kbps)']]
        
        # Normalizing the data
        scaler = MinMaxScaler()
        normalized_metrics = scaler.fit_transform(experience_metrics)
        
        # Applying KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        agg_data['Cluster'] = kmeans.fit_predict(normalized_metrics)
        
        # 2D Plotting the clusters
        plt.figure(figsize=(12, 8))
        
        # Plot for TCP Retransmission vs. Avg RTT
        plt.subplot(1, 2, 1)
        sns.scatterplot(x='Average TCP Retrans. Vol (Bytes)', y='Avg RTT (ms)', hue='Cluster', data=agg_data, palette='viridis', s=100, alpha=0.7)
        plt.title('TCP Retransmission vs. Avg RTT')
        
        # Plot for Avg Bearer TP vs. Avg RTT
        plt.subplot(1, 2, 2)
        sns.scatterplot(x='Avg Bearer TP (kbps)', y='Avg RTT (ms)', hue='Cluster', data=agg_data, palette='viridis', s=100, alpha=0.7)
        plt.title('Avg Bearer TP vs. Avg RTT')
        
        plt.tight_layout()
        plt.show()
        
        # 3D Plotting the clusters
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plotting 3D scatter plot for all three metrics
        scatter = ax.scatter(
            agg_data['Average TCP Retrans. Vol (Bytes)'], 
            agg_data['Avg RTT (ms)'], 
            agg_data['Avg Bearer TP (kbps)'], 
            c=agg_data['Cluster'], cmap='viridis', s=100, alpha=0.7
        )
        
        # Setting axis labels
        ax.set_xlabel('Average TCP Retrans. Vol (Bytes)')
        ax.set_ylabel('Avg RTT (ms)')
        ax.set_zlabel('Avg Bearer TP (kbps)')
        
        plt.title('3D Cluster Plot: TCP Retransmission, Avg RTT, and Avg Bearer TP')
        plt.colorbar(scatter)
        plt.show()
        
        # Describing each cluster
        # cluster_summary = agg_data.groupby('Cluster')
        
        # return cluster_summary
        