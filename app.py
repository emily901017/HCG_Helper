"""
HGC Helper - Streamlit Application
Main application with student chat interface and teacher dashboard
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import uuid
from engine import RAGEngine
from database import QueryLogger
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Page configuration
st.set_page_config(
    page_title="HGC Helper",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_engine" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        st.session_state.rag_engine = RAGEngine()


def student_interface():
    """Student chat interface"""
    st.title("📚 HGC Helper - Your Learning Companion")
    st.markdown("Ask me anything about History, Geography, or Civics!")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_engine.query(
                    question=prompt,
                    session_id=st.session_state.session_id
                )
                st.markdown(response)

        # Add assistant response to chat
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info(
            "HGC Helper is an AI-powered tutor for high school social studies. "
            "Ask questions about History, Geography, and Civics, and get accurate "
            "answers based on your textbooks."
        )

        st.header("Tips for Better Questions")
        st.markdown("""
        - Be specific about what you want to know
        - Mention the subject or topic if relevant
        - Ask one question at a time
        - Use proper terminology when possible
        """)

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


def teacher_dashboard():
    """Teacher dashboard for viewing student queries"""
    st.title("👨‍🏫 Teacher Dashboard")
    st.markdown("Analyze student questions to identify learning difficulties")

    query_logger = QueryLogger()

    # Get all queries
    queries = query_logger.get_all_queries()
    stats = query_logger.get_query_statistics()

    # Statistics overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Questions", stats['total_queries'])

    with col2:
        st.metric("Questions (Last 7 Days)", stats['recent_queries_7days'])

    with col3:
        unique_sessions = len(set(q['session_id'] for q in queries if q['session_id']))
        st.metric("Unique Sessions", unique_sessions)

    # Subject distribution
    if stats['by_subject']:
        st.subheader("Questions by Subject")
        subject_df = pd.DataFrame(
            list(stats['by_subject'].items()),
            columns=['Subject', 'Count']
        )
        fig = px.bar(subject_df, x='Subject', y='Count', color='Subject')
        st.plotly_chart(fig, use_container_width=True)

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["All Questions", "Search", "Insights"])

    with tab1:
        st.subheader("All Student Questions")

        if queries:
            # Convert to DataFrame
            df = pd.DataFrame(queries)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Display options
            col1, col2 = st.columns(2)
            with col1:
                sort_by = st.selectbox(
                    "Sort by",
                    ["timestamp", "question"],
                    index=0
                )
            with col2:
                ascending = st.checkbox("Ascending", value=False)

            # Sort and display
            df_display = df.sort_values(by=sort_by, ascending=ascending)
            df_display['timestamp'] = df_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

            st.dataframe(
                df_display[['timestamp', 'question', 'session_id']],
                use_container_width=True,
                hide_index=True
            )

            # Download option
            csv = df_display.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"student_queries_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No questions logged yet.")

    with tab2:
        st.subheader("Search Questions")

        search_term = st.text_input("Enter keyword to search")

        if search_term:
            results = query_logger.search_queries(search_term)

            if results:
                st.success(f"Found {len(results)} matching questions")
                results_df = pd.DataFrame(results)
                results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
                results_df['timestamp'] = results_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

                st.dataframe(
                    results_df[['timestamp', 'question']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("No matching questions found.")

    with tab3:
        st.subheader("Question Insights")

        if queries:
            # Word cloud of common keywords
            st.write("**Most Common Keywords in Questions**")

            keywords = query_logger.get_common_keywords(limit=50)

            if keywords:
                # Display top keywords as a table
                keywords_df = pd.DataFrame(keywords, columns=['Keyword', 'Frequency'])
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.dataframe(keywords_df.head(20), use_container_width=True, hide_index=True)

                with col2:
                    # Create word cloud
                    wordcloud_dict = dict(keywords)
                    if wordcloud_dict:
                        wordcloud = WordCloud(
                            width=400,
                            height=300,
                            background_color='white'
                        ).generate_from_frequencies(wordcloud_dict)

                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)

            # Question frequency over time
            st.write("**Question Frequency Over Time**")
            df = pd.DataFrame(queries)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date

            daily_counts = df.groupby('date').size().reset_index(name='count')

            fig = px.line(
                daily_counts,
                x='date',
                y='count',
                title='Questions Per Day',
                labels={'date': 'Date', 'count': 'Number of Questions'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available yet.")

    # Sidebar
    with st.sidebar:
        st.header("Dashboard Info")
        st.info(
            "This dashboard provides insights into student questions, helping "
            "you identify areas where students need more support."
        )

        if st.button("Refresh Data"):
            st.rerun()


def main():
    """Main application entry point"""
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Interface",
        ["Student Chat", "Teacher Dashboard"]
    )

    # Route to appropriate interface
    if page == "Student Chat":
        student_interface()
    else:
        teacher_dashboard()


if __name__ == "__main__":
    main()
