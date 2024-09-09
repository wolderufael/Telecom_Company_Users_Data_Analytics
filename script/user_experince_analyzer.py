class ExperienceAnalyzer:
    def user_aggregate(self,data):
        agg_data = data.groupby('MSISDN/Number').agg({
            'Avg Bearer TP DL (kbps)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Avg RTT UL (ms)': 'mean',
            'Handset Type': 'first'
        }).reset_index()

        print(agg_data.head())
        
    def top_bottom_frequent(self,data):
        agg_data = data.groupby('MSISDN/Number').agg({
            'Avg Bearer TP DL (kbps)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Avg RTT UL (ms)': 'mean',
            'Handset Type': 'first'
        }).reset_index()
        
        # Top, Bottom, and Most Frequent TCP values
        top_tcp = agg_data['Avg Bearer TP DL (kbps)'].nlargest(10)
        bottom_tcp = agg_data['Avg Bearer TP DL (kbps)'].nsmallest(10)
        most_frequent_tcp = agg_data['Avg Bearer TP DL (kbps)'].value_counts().nlargest(10)
        # Same can be done for RTT and throughput
        top_rtt = agg_data['Avg RTT DL (ms)'].nlargest(10)
        bottom_rtt = agg_data['Avg RTT DL (ms)'].nsmallest(10)
        most_frequent_rtt = agg_data['Avg RTT DL (ms)'].value_counts().nlargest(10)