import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

@st.cache_data # to load data much faster
def load_data():
    # load csv files which gives us info about videos, subscribers, performance
    df_agg = pd.read_csv("E:/Projects/Youtube-Analytics-Dashboard/data/Aggregated_Metrics_By_Video.csv").iloc[1:,:]
    df_subsrcibers = pd.read_csv("E:/Projects/Youtube-Analytics-Dashboard/data/Aggregated_Metrics_By_Country_And_Subscriber_Status.csv")
    df_comments =pd.read_csv("E:/Projects/Youtube-Analytics-Dashboard/data/All_Comments_Final.csv")
    df_time = pd.read_csv("E:/Projects/Youtube-Analytics-Dashboard/data/Video_Performance_Over_Time.csv")

    # 
    newcols =[x.encode("ascii", "ignore").decode('utf-8') for x in df_agg.columns]
    df_agg.columns = newcols

    #convert to date time format for df_video dataFrame
    df_agg['Video publish time'] = pd.to_datetime(df_agg['Video publish time'], format='mixed' )
    df_agg['Average view duration'] = df_agg['Average view duration'].apply(lambda x : datetime.strptime(x, '%H:%M:%S'))

    df_agg['Avg_duration_sec'] = df_agg['Average view duration'].apply(lambda x: x.second + x.minute*60 + x.hour*3600)
    df_agg['Engagement_ratio'] = (df_agg['Comments added'] + df_agg['Shares'] + df_agg['Likes'] +df_agg['Dislikes']) / df_agg['Views']
    df_agg['Views / sub gained'] = df_agg['Views'] / df_agg['Subscribers gained']
    df_agg.sort_values('Video publish time', ascending=False, inplace= True)
    #convert to date time format for df_time dataFrame
    df_time['Date'] = pd.to_datetime(df_time['Date'], format='mixed')

    return df_agg, df_subsrcibers, df_comments, df_time

def style_red(v, props=''):
    """Styyle negative numerics"""
    try:
        return props if v < 0 else None
    except:
        ValueError("Could not style numericcal value:",v)
def style_green(v, props=''):
    """Styyle positive numerics"""
    try:
        return props if v > 0 else None
    except:
        ValueError("Could not style numericcal value:",v)

def audience_sample(country):
    """Top countries"""
    if country == 'US':
        return 'USA'
    if country == 'IN':
        return 'India'
    else:
        return "Other"


#load DataFrames from function load_data()
df_agg, df_subsrcibers, df_comments, df_time = load_data()

#fEATURE engineerirng
df_agg_diff = df_agg.copy()
metric_date_12month  = df_agg_diff['Video publish time'].max() - pd.DateOffset(months= 12)
median_agg = df_agg_diff[df_agg_diff['Video publish time'] >= metric_date_12month].median(numeric_only = True)
numeric_cols = np.array((df_agg_diff.dtypes == 'float64') | (df_agg_diff.dtypes == 'int64'))
df_agg_diff.iloc[:,numeric_cols] = (df_agg_diff.iloc[:,numeric_cols] - median_agg) / (median_agg)

#build dashboard
add_sidebar = st.sidebar.selectbox('Aggregate or Individual video', ('Aggregate Metrics', 'Individual Video Analysis'))

if add_sidebar == 'Aggregate Metrics':
    df_agg_metrics = df_agg[['Video publish time', 'Views', 'Likes', 'Subscribers', 'Shares', 'Comments added', 'RPM (USD)', 'Average percentage viewed (%)',
                                 'Avg_duration_sec', 'Engagement_ratio', 'Views / sub gained']]
    metric_date_6month = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months =6)
    metric_date_12month = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months =12)
    metric_median_6months = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_6month].median(numeric_only = True)
    metric_median_12months = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_12month].median(numeric_only = True)

    # st.metric('Views', metric_median_6months['Views'], 500)
    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]
    count =0

    for i in metric_median_6months.index:
        with columns[count]:
            delta = (metric_median_6months[i] - metric_median_12months[i]) / metric_median_12months[i]
            st.metric(label= i, value = round(metric_median_6months[i],1), delta= "{:.2%}".format(delta))
            count +=1
            if count>= 5:
                count =0

    df_agg_diff['Publish_date'] = df_agg_diff['Video publish time'].apply(lambda x: x.date())
    df_agg_diff_final = df_agg_diff.loc[:, ['Video title', 'Publish_date', 'Views', 'Likes', 'Subscribers', 'Shares', 'Comments added', 'RPM (USD)', 'Average percentage viewed (%)',
                                 'Avg_duration_sec', 'Engagement_ratio', 'Views / sub gained']]
    #Sorted by Views from highest -> lowest
    df_agg_diff_final.sort_values(by=['Views'], ascending =False, inplace =True)

    #format to %
    df_agg_numeric_list = df_agg_diff_final.median(numeric_only= True).index.tolist()
    df_pct ={}
    for i in df_agg_numeric_list:
        df_pct[i] = '{:.1%}'.format
    
    #write eveything to streamlit dashboard
    st.dataframe(df_agg_diff_final.style.map(style_red, props= 'color:red;').map(style_green, props = 'color:green;').format(df_pct)) # type: ignore

if add_sidebar == 'Individual Video Analysis':
    videos = tuple(df_agg['Video title'])
    video_select = st.selectbox('Video picker', videos)

    agg_filtered = df_agg[df_agg['Video title'] == video_select]
    agg_sub_filtered = df_subsrcibers[df_subsrcibers['Video Title'] == video_select]
    agg_sub_filtered['Country'] = agg_sub_filtered['Country Code'].apply(audience_sample)
    agg_sub_filtered.sort_values(by= ['Is Subscribed'], inplace =True)

    fig = px.bar(agg_sub_filtered, x= 'Views', y= 'Is Subscribed', color= 'Country', pattern_shape="Country", pattern_shape_sequence=[".", "x", "+"])
    st.plotly_chart(fig)
