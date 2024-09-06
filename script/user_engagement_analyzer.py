import pandas as pd

class EngagementAnalyzer:

    def aggregate_engagement_metrics(self, data):
        # Step 1: Aggregate session metrics per customer (MSISDN)
        customer_aggregation = data.groupby('MSISDN/Number').agg(
            session_frequency=('Bearer Id', 'count'),  # Number of sessions
            total_duration=('Dur. (ms)', 'sum'),  # Total session duration (in milliseconds)
            total_traffic=('Total DL (Bytes)', 'sum') + data['Total UL (Bytes)'].sum()  # Total traffic (download + upload)
        ).reset_index()
        
        # Step 2: Sort and report top 10 customers for each metric
        
        # Top 10 customers by session frequency
        top_10_session_frequency = customer_aggregation.nlargest(10, 'session_frequency')[['MSISDN/Number', 'session_frequency']]
        
        # Top 10 customers by total session duration
        top_10_session_duration = customer_aggregation.nlargest(10, 'total_duration')[['MSISDN/Number', 'total_duration']]
        
        # Top 10 customers by total traffic (download + upload)
        top_10_total_traffic = customer_aggregation.nlargest(10, 'total_traffic')[['MSISDN/Number', 'total_traffic']]
        
        return top_10_session_frequency, top_10_session_duration, top_10_total_traffic

    # Assuming you have your raw data in `data`
    # You can now call this function to get the top 10 customers per engagement metric
    # top_10_freq, top_10_duration, top_10_traffic = instance.aggregate_engagement_metrics(data)
