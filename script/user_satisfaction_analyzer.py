from scipy.spatial.distance import euclidean


class SatisfactionAnalyzer:
  
    def calculate_engagement_experience_scores(agg_data, cluster_column='Cluster', low_engaged_cluster=0, worst_experience_cluster=0):
        engagement_scores = []
        experience_scores = []
        
        # Calculate Euclidean distance for each user
        for i, row in agg_data.iterrows():
            engagement_score = euclidean(
                row[['Average TCP Retrans. Vol (Bytes)', 'Avg RTT (ms)', 'Avg Bearer TP (kbps)']], 
                agg_data[agg_data[cluster_column] == low_engaged_cluster].iloc[0][['Average TCP Retrans. Vol (Bytes)', 'Avg RTT (ms)', 'Avg Bearer TP (kbps)']]
            )
            experience_score = euclidean(
                row[['Average TCP Retrans. Vol (Bytes)', 'Avg RTT (ms)', 'Avg Bearer TP (kbps)']], 
                agg_data[agg_data[cluster_column] == worst_experience_cluster].iloc[0][['Average TCP Retrans. Vol (Bytes)', 'Avg RTT (ms)', 'Avg Bearer TP (kbps)']]
            )
            engagement_scores.append(engagement_score)
            experience_scores.append(experience_score)
        
        agg_data['Engagement Score'] = engagement_scores
        agg_data['Experience Score'] = experience_scores
        
        return agg_data
