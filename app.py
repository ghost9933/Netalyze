# Import libraries
import re
import streamlit as st
import pandas as pd
import tempfile
from zipfile import ZipFile
import shutil
from thefuzz import process
import plotly.express as px
import networkx as nx
from pyvis.network import Network
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import nltk
import os
import base64

from io import BytesIO

# Download NLTK stopwords if not already downloaded
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# Set Streamlit page configuration
st.set_page_config(
    page_title="LinkedIn Network Visualizer",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Helper function to find the header row dynamically
def find_header_row(file_path, header_keywords):
    """Finds the header row in the CSV file by searching for specific keywords."""
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if all(keyword.lower() in line.lower() for keyword in header_keywords):
                return i
    return -1

# Helper functions
def get_data(usr_file, data="connections") -> pd.DataFrame:
    """Extracts CSV data from a zip file."""
    if usr_file is None:
        return None

    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            with ZipFile(usr_file, "r") as zipObj:
                zipObj.extractall(tmpdirname)
        except Exception as e:
            st.error(f"Error extracting ZIP file: {e}")
            return None

        if data == "connections":
            file_path = os.path.join(tmpdirname, "Connections.csv")
            header_keywords = ["First Name", "Last Name", "Company", "Position", "Connected On"]
        elif data == "messages":
            file_path = os.path.join(tmpdirname, "messages.csv")
            header_keywords = ["Conversation ID", "Conversation Title", "From", "Subject", "Content"]
        else:
            st.error("Invalid data type specified.")
            return None

        if not os.path.isfile(file_path):
            st.error(f"{'Connections' if data == 'connections' else 'Messages'} file not found in the ZIP.")
            return None

        try:
            header_row = find_header_row(file_path, header_keywords)
            if header_row == -1:
                st.error(f"Header row not found in {os.path.basename(file_path)}.")
                return None

            # Read the CSV from the header row
            df = pd.read_csv(file_path, skiprows=header_row)
            st.write(f"Successfully loaded `{os.path.basename(file_path)}` with columns: {list(df.columns)}")
            return df
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None


def clean_df(df: pd.DataFrame, privacy: bool = False) -> pd.DataFrame:
    """Cleans the dataframe containing LinkedIn connections data."""
    required_columns = ["first_name", "last_name", "email_address", "company", "position", "connected_on"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Missing columns in data: {', '.join(missing_columns)}")
        st.write(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()

    if privacy:
        df = df.drop(columns=["first_name", "last_name", "email_address"], errors='ignore')

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Drop missing values in company and position
    df = df.dropna(subset=["company", "position"])

    # Combine first name and last name if present
    if "first_name" in df.columns and "last_name" in df.columns:
        df['name'] = df['first_name'].astype(str) + ' ' + df['last_name'].astype(str)
        df = df.drop(columns=["first_name", "last_name"], errors='ignore')

    # Truncate company names
    df['company'] = df['company'].astype(str).str[:35]

    # Convert 'connected_on' to datetime
    df['connected_on'] = pd.to_datetime(df['connected_on'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=["connected_on"])

    # Filter out unwanted companies
    df = df[~df['company'].str.contains(r"[Ff]reelance|[Ss]elf-[Ee]mployed|\.|\-", regex=True)]

    # Fuzzy match for positions
    replace_fuzzywuzzy_match(df, "position", "Data Scientist")
    replace_fuzzywuzzy_match(df, "position", "Software Engineer", min_ratio=85)

    st.write(f"Cleaned data has {len(df)} records after processing.")
    return df


def replace_fuzzywuzzy_match(df: pd.DataFrame, column: str, query: str, min_ratio: int = 75):
    """Replaces fuzzy matches in the specified column with the query string."""
    pos_names = df[column].unique()
    matches = process.extract(query, pos_names, limit=500)
    matching_pos_name = [match[0] for match in matches if match[1] >= min_ratio]
    matches_rows = df[column].isin(matching_pos_name)
    df.loc[matches_rows, column] = query


def agg_sum(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Aggregates value counts for companies and positions."""
    agg_df = df[name].value_counts().reset_index()
    agg_df.columns = [name, "count"]
    agg_df = agg_df.sort_values(by="count", ascending=False)
    return agg_df


def plot_bar(df: pd.DataFrame, rows: int, title=""):
    """Creates a horizontal bar plot for the top N entries in the dataframe."""
    fig = px.bar(
        df.head(rows),
        x='count',
        y=df.columns[0],
        orientation='h',
        template="plotly_dark",
        labels={df.columns[0]: title.split(" ")[1], "count": "Count"},
        title=title
    )
    fig.update_layout(
        height=400,
        width=600,
        margin=dict(pad=5),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_timeline(df: pd.DataFrame):
    """Generates a timeline plot of connections over time."""
    df_time = df["connected_on"].dt.to_period('M').dt.to_timestamp()
    timeline_df = df_time.value_counts().reset_index()
    timeline_df.columns = ["connected_on", "count"]
    timeline_df = timeline_df.sort_values(by="connected_on")

    fig = px.line(
        timeline_df,
        x="connected_on",
        y="count",
        title="Timeline of Connections",
        template="plotly_dark",
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Connections",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]),
                bgcolor="black",
            ),
            rangeslider=dict(visible=True),
            type="date",
        ),
    )
    return fig


def plot_wordcloud(chats: pd.DataFrame):
    """Generates a word cloud from chat messages."""
    required_columns = ["subject", "content"]
    if not all(col in chats.columns for col in required_columns):
        st.warning("Messages data does not contain required columns for word cloud.")
        return None

    chats_nospam = chats[chats["subject"].isnull()]
    chats_nospam_nohtml = chats_nospam[~chats_nospam["content"].str.contains("<|>", na=False)]
    messages = chats_nospam_nohtml.dropna(subset=["content"])["content"].astype(str).values
    corpus = []

    for message in messages:
        message = re.sub(r"http[s]?\S+", "", message)  # Remove URLs
        message = re.sub("[^a-zA-Z]", " ", message).lower()  # Keep only letters
        message = re.sub(r"\s+[a-zA-Z]\s+", " ", message)  # Remove single letters
        words = [word for word in message.split() if word not in set(stopwords.words("english"))]
        corpus.append(" ".join(words))

    combined_text = " ".join(corpus)

    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="black",
        stopwords=STOPWORDS,
        max_words=200,
        colormap="viridis",
    ).generate(combined_text)

    # Plot word cloud
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    return plt


def plot_network_graph(df: pd.DataFrame):
    """Generates an interactive network graph of connections."""
    G = nx.Graph()

    # Adding nodes
    for _, row in df.iterrows():
        G.add_node(row['name'], company=row['company'], position=row['position'])

    # Adding edges (for simplicity, connecting all to a central node, e.g., you)
    central_node = "You"
    G.add_node(central_node, company="You", position="Your Position")
    for node in G.nodes():
        if node != central_node:
            G.add_edge(central_node, node)

    # Generate PyVis network
    net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white')
    net.from_nx(G)

    # Customize node appearance
    for node in net.nodes:
        node['title'] = f"Company: {G.nodes[node['id']]['company']}<br>Position: {G.nodes[node['id']]['position']}"
        if node['id'] == central_node:
            node['color'] = 'red'
            node['size'] = 20
        else:
            node['color'] = 'blue'
            node['size'] = 10

    return net


def plot_sankey(df: pd.DataFrame):
    """Generates a Sankey diagram showing flow between companies and positions."""
    sankey_df = df.groupby(['company', 'position']).size().reset_index(name='count')

    # Create list of unique labels
    companies = sankey_df['company'].unique().tolist()
    positions = sankey_df['position'].unique().tolist()
    labels = companies + positions

    # Create mapping for sources and targets
    source_indices = sankey_df['company'].apply(lambda x: labels.index(x)).tolist()
    target_indices = sankey_df['position'].apply(lambda x: labels.index(x)).tolist()

    fig = px.sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=["blue"]*len(companies) + ["green"]*len(positions)
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=sankey_df['count']
        ),
        title="Flow from Companies to Positions",
    )
    fig.update_layout(template="plotly_dark")
    return fig


def add_bg_from_local(image_file):
    """Adds a background image to the Streamlit app."""
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded_string});
        background-size: cover;
        background-position: center;
    }}
    </style>
    """,
        unsafe_allow_html=True
    )


def main():
    """Main function to run the Streamlit app."""
    # Optional: Add a background image
    # add_bg_from_local('path_to_background_image.png')  # Uncomment and set the correct path

    st.markdown(
        """
        <h1 style='text-align: center; color: white;'>🔗 LinkedIn Network Visualizer</h1>
        <h3 style='text-align: center; color: white;'>Unlock Insights from Your LinkedIn Connections</h3>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Upload Your LinkedIn Data")
    st.sidebar.markdown("Upload a ZIP file containing `Connections.csv` and optionally `messages.csv`.")

    # Upload files
    usr_file = st.sidebar.file_uploader("📁 Upload ZIP File", type={"zip"})

    if usr_file is not None:
        df_connections = get_data(usr_file, data="connections")
        df_messages = get_data(usr_file, data="messages")

        if df_connections is None or df_connections.empty:
            st.error("Connections data is empty or invalid.")
            return

        df_clean = clean_df(df_connections)

        if df_clean.empty:
            st.error("Cleaned data is empty. Please check your data and try again.")
            return

        with st.expander("🔍 Show Raw Connections Data"):
            st.dataframe(df_connections)

        if df_messages is not None and not df_messages.empty:
            with st.expander("💬 Show Raw Messages Data"):
                st.dataframe(df_messages)

        # Data wrangling
        agg_df_company = agg_sum(df_clean, "company")
        agg_df_position = agg_sum(df_clean, "position")

        total_conn = len(df_clean)

        # Display metrics
        col1, col2 = st.columns(2)
        col1.metric("Total Connections", total_conn)
        if not agg_df_company.empty:
            col2.metric("Top Company", f"{agg_df_company['company'].iloc[0]} ({agg_df_company['count'].iloc[0]})")

        # Sidebar filters
        st.sidebar.header("Filters")
        top_n = st.sidebar.slider("Select Top N", 5, 50, 10)

        # Visualizations
        st.subheader("📊 Top Companies & Positions")
        bar_col1, bar_col2 = st.columns(2)
        with bar_col1:
            if not agg_df_company.empty:
                company_plt = plot_bar(agg_df_company, top_n, title="Top Companies")
                st.plotly_chart(company_plt, use_container_width=True)
            else:
                st.write("No company data available.")

        with bar_col2:
            if not agg_df_position.empty:
                position_plt = plot_bar(agg_df_position, top_n, title="Top Positions")
                st.plotly_chart(position_plt, use_container_width=True)
            else:
                st.write("No position data available.")

        st.subheader("⏰ Timeline of Connections")
        timeline_fig = plot_timeline(df_clean)
        st.plotly_chart(timeline_fig, use_container_width=True)

        if df_messages is not None and not df_messages.empty:
            st.subheader("☁️ Word Cloud of Chats")
            wordcloud_fig = plot_wordcloud(df_messages)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)

        st.subheader("🔗 Network Graph of Connections")
        network_graph = plot_network_graph(df_clean)
        network_graph_html = network_graph.generate_html()
        st.components.v1.html(network_graph_html, height=750, width='100%')

        st.subheader("🔄 Sankey Diagram: Companies to Positions")
        sankey_fig = plot_sankey(df_clean)
        st.plotly_chart(sankey_fig, use_container_width=True)

    else:
        st.info("Please upload a ZIP file containing your LinkedIn data to get started.")

    # Footer
    st.markdown(
        """
        <hr>
        <p style='text-align: center; color: gray;'>
            Developed with ❤️ using Streamlit
        </p>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
