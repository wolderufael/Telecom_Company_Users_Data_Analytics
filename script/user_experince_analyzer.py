import matplotlib.pyplot as plt
import seaborn as sns

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
