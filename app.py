import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns

st.sidebar.title("Whatsapp Chat Analyzer")
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data=bytes_data.decode("utf-8")
    df=preprocessor.preprocess(data)



    user_list = df['user'].unique().tolist()
    if 'group_notifications' in user_list:
        user_list.remove('group_notifications')

    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        # Monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)

        if timeline.empty:
            st.warning("No messages available for timeline analysis.")
        else:
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Weekly Activity Map
        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)

        if user_heatmap.empty:
            st.warning("No activity data available for heatmap.")
        else:
            fig, ax = plt.subplots()
            sns.heatmap(user_heatmap, cmap="coolwarm", annot=True)
            st.pyplot(fig)


        # Sentiment Analysis
        st.title("Sentiment Analysis")
        sentiment_df = helper.sentiment_analysis(selected_user, df)

        if sentiment_df.empty:
            st.warning("No sentiment data available.")
        else:
            fig, ax = plt.subplots()
            ax.hist(sentiment_df['sentiment'], bins=20, color='blue', alpha=0.7)
            plt.xlabel('Sentiment Score')
            plt.ylabel('Message Count')
            plt.title('Sentiment Distribution')
            st.pyplot(fig)

        sentiment_df['sentiment_label'] = sentiment_df['sentiment'].apply(lambda x:
                                                                          'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

        # Count the occurrences of each category
        sentiment_counts = sentiment_df['sentiment_label'].value_counts()

        # Plot Pie Chart
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
               colors=['green', 'red', 'gray'], startangle=90, shadow=True)
        plt.title('Sentiment Distribution')
        st.pyplot(fig)



