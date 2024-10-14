# Import necessary libraries
import re
import streamlit as st
import pandas as pd
import tempfile
from zipfile import ZipFile
import shutil
from thefuzz import process
import plotly.express as px
import plotly.graph_objects as go
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
    """
    Finds the header row in the CSV file by searching for specific keywords.
    
    Parameters:
    - file_path (str): Path to the CSV file.
    - header_keywords (list): List of keywords expected in the header row.
    
    Returns:
    - int: The row index of the header row. Returns -1 if not found.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if all(keyword.lower() in line.lower() for keyword in header_keywords):
                return i
    return -1

# Helper function to extract and load data from ZIP
def get_data(usr_file, data_type="connections") -> pd.DataFrame:
    """
    Extracts CSV data from a ZIP file and loads it into a DataFrame.
    
    Parameters:
    - usr_file: Uploaded ZIP file from Streamlit.
    - data_type (str): Type of data to extract ('connections' or 'messages').
    
    Returns:
    - pd.DataFrame: Loaded DataFrame or None if an error occurs.
    """
    if usr_file is None:
        return None

    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            with ZipFile(usr_file, "r") as zipObj:
                zipObj.extractall(tmpdirname)
        except Exception as e:
            st.error(f"Error extracting ZIP file: {e}")
            return None

        # Define file paths and header keywords based on data type
        if data_type == "connections":
            file_name = "Connections.csv"
            header_keywords = ["First Name", "Last Name", "Company", "Position", "Connected On"]
        elif data_type == "messages":
            file_name = "messages.csv"
            header_keywords = ["Conversation ID", "Conversation Title", "From", "Subject", "Content"]
        else:
            st.error("Invalid data type specified.")
            return None

        file_path = os.path.join(tmpdirname, file_name)

        if not os.path.isfile(file_path):
            st.error(f"{file_name} not found in the ZIP.")
            return None

        try:
            header_row = find_header_row(file_path, header_keywords)
            if header_row == -1:
                st.error(f"Header row not found in {file_name}. Please ensure the CSV has the correct format.")
                return None

            # Read the CSV from the header row
            df = pd.read_csv(file_path, skiprows=header_row, header=0)
            st.write(f"✅ Successfully loaded `{file_name}` with columns: {list(df.columns)}")
            return df
        except Exception as e:
            st.error(f"Error reading {file_name}: {e}")
            return None

# Helper function to clean connections DataFrame
def clean_connections_df(df: pd.DataFrame, privacy: bool = False) -> pd.DataFrame:
    """
    Cleans the DataFrame containing LinkedIn connections data.
    
    Parameters:
    - df (pd.DataFrame): Raw connections DataFrame.
    - privacy (bool): If True, removes sensitive information.
    
    Returns:
    - pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=True)

    # Define required columns
    required_columns = ["first_name", "last_name", "email_address", "company", "position", "connected_on"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"🚫 Missing columns in data: {', '.join(missing_columns)}")
        st.write(f"🔍 Available columns: {list(df.columns)}")
        return pd.DataFrame()

    # Remove sensitive columns if privacy is enabled
    if privacy:
        df.drop(columns=["first_name", "last_name", "email_address"], inplace=True, errors='ignore')

    # Drop rows with missing company or position
    df.dropna(subset=["company", "position"], inplace=True)

    # Combine first name and last name into a single 'name' column
    if "first_name" in df.columns and "last_name" in df.columns:
        df['name'] = df['first_name'].astype(str) + ' ' + df['last_name'].astype(str)
        df.drop(columns=["first_name", "last_name"], inplace=True, errors='ignore')

    # Truncate company names to 35 characters
    df['company'] = df['company'].astype(str).str[:35]

    # Convert 'connected_on' to datetime, ensuring day-first format
    df['connected_on'] = pd.to_datetime(df['connected_on'], dayfirst=True, errors='coerce')
    df.dropna(subset=["connected_on"], inplace=True)

    # Filter out unwanted companies (e.g., Freelance, Self-Employed)
    df = df[~df['company'].str.contains(r"(?i)\bFreelance\b|\bSelf-Employed\b", regex=True, na=False)]

    # Fuzzy match and replace position titles
    replace_fuzzywuzzy_match(df, "position", "Data Scientist")
    replace_fuzzywuzzy_match(df, "position", "Software Engineer", min_ratio=85)

    st.write(f"✨ Cleaned data has {len(df)} records after processing.")
    return df

# Helper function to clean messages DataFrame
def clean_messages_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame containing LinkedIn messages data.
    
    Parameters:
    - df (pd.DataFrame): Raw messages DataFrame.
    
    Returns:
    - pd.DataFrame: Cleaned DataFrame.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_', regex=True)

    # Define required columns
    required_columns = ["conversation_id", "conversation_title", "from", "subject", "content"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"🚫 Missing columns in messages data: {', '.join(missing_columns)}")
        st.write(f"🔍 Available columns in messages: {list(df.columns)}")
        return pd.DataFrame()

    # Filter out spam messages or irrelevant content (e.g., folders other than 'Inbox')
    if 'folder' in df.columns:
        df = df[df['folder'].str.lower() == 'inbox']
    else:
        st.warning("⚠️ 'folder' column not found in messages data. Skipping folder-based filtering.")

    # Drop rows with missing content
    df.dropna(subset=["content"], inplace=True)

    st.write(f"✨ Cleaned messages data has {len(df)} records after processing.")
    return df

# Helper function to replace fuzzy matches
def replace_fuzzywuzzy_match(df: pd.DataFrame, column: str, query: str, min_ratio: int = 75):
    """
    Replaces fuzzy matches in the specified column with the query string.
    
    Parameters:
    - df (pd.DataFrame): DataFrame to process.
    - column (str): Column name to perform fuzzy matching on.
    - query (str): The string to replace matched values with.
    - min_ratio (int): Minimum matching ratio to consider a match.
    """
    pos_names = df[column].unique()
    matches = process.extract(query, pos_names, limit=500)
    matching_pos_name = [match[0] for match in matches if match[1] >= min_ratio]
    matches_rows = df[column].isin(matching_pos_name)
    df.loc[matches_rows, column] = query

# Helper function to aggregate counts
def agg_sum(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Aggregates value counts for a specified column.
    
    Parameters:
    - df (pd.DataFrame): DataFrame to aggregate.
    - column (str): Column name to count unique values.
    
    Returns:
    - pd.DataFrame: Aggregated counts.
    """
    agg_df = df[column].value_counts().reset_index()
    agg_df.columns = [column, "count"]
    agg_df = agg_df.sort_values(by="count", ascending=False)
    return agg_df

# Helper function to plot bar charts
def plot_bar(df: pd.DataFrame, top_n: int, title: str) -> px.bar:
    """
    Creates a horizontal bar plot for the top N entries in the dataframe.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with aggregated counts.
    - top_n (int): Number of top entries to display.
    - title (str): Title of the plot.
    
    Returns:
    - plotly.graph_objects.Figure: Plotly bar chart.
    """
    fig = px.bar(
        df.head(top_n),
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

# Helper function to plot timeline
def plot_timeline(df: pd.DataFrame) -> px.line:
    """
    Generates a timeline plot of connections over time.
    
    Parameters:
    - df (pd.DataFrame): Cleaned connections DataFrame.
    
    Returns:
    - plotly.graph_objects.Figure: Plotly line chart.
    """
    df_time = df["connected_on"].dt.to_period('M').dt.to_timestamp()
    timeline_df = df_time.value_counts().reset_index()
    timeline_df.columns = ["connected_on", "count"]
    timeline_df = timeline_df.sort_values(by="connected_on")
    
    fig = px.line(
        timeline_df,
        x="connected_on",
        y="count",
        title="⏰ Timeline of Connections",
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

# Helper function to plot word cloud
def plot_wordcloud(df: pd.DataFrame) -> plt.Figure:
    """
    Generates a word cloud from chat messages.
    
    Parameters:
    - df (pd.DataFrame): Cleaned messages DataFrame.
    
    Returns:
    - matplotlib.figure.Figure: Matplotlib figure containing the word cloud.
    """
    # Ensure required columns are present
    required_columns = ["subject", "content"]
    if not all(col in df.columns for col in required_columns):
        st.warning("⚠️ Messages data does not contain required columns for word cloud.")
        return None

    # Filter out spam or irrelevant messages
    chats_nospam = df[df["subject"].isnull()]
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

    # Generate word cloud with mask if available
    try:
        linkedin_mask = np.array(Image.open("media/linkedin.png"))
    except FileNotFoundError:
        st.warning("⚠️ Mask image `media/linkedin.png` not found. Using default settings for word cloud.")
        linkedin_mask = None

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="black",
        stopwords=STOPWORDS,
        mask=linkedin_mask,
        contour_color="white",
        contour_width=2,
        colormap="viridis",
    ).generate(combined_text)

    # Plot word cloud
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    return plt

# Helper function to plot network graph
def plot_network_graph(df: pd.DataFrame) -> Network:
    """
    Generates an interactive network graph of connections.
    
    Parameters:
    - df (pd.DataFrame): Cleaned connections DataFrame.
    
    Returns:
    - pyvis.network.Network: PyVis network graph object.
    """
    G = nx.Graph()

    # Add central node
    central_node = "You"
    G.add_node(central_node, company="You", position="Your Position")

    # Add connection nodes and edges
    for _, row in df.iterrows():
        G.add_node(row['name'], company=row['company'], position=row['position'])
        G.add_edge(central_node, row['name'])

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

# Helper function to plot Sankey diagram using plotly.graph_objects
def plot_sankey_go(df: pd.DataFrame) -> go.Figure:
    """
    Generates a Sankey diagram showing flow between companies and positions using plotly.graph_objects.
    
    Parameters:
    - df (pd.DataFrame): Cleaned connections DataFrame.
    
    Returns:
    - plotly.graph_objects.Figure: Plotly Sankey diagram.
    """
    sankey_df = df.groupby(['company', 'position']).size().reset_index(name='count')
    
    # Create list of unique labels
    companies = sankey_df['company'].unique().tolist()
    positions = sankey_df['position'].unique().tolist()
    labels = companies + positions
    
    # Create mapping for sources and targets
    source_indices = sankey_df['company'].apply(lambda x: labels.index(x)).tolist()
    target_indices = sankey_df['position'].apply(lambda x: labels.index(x)).tolist()
    
    link = dict(source=source_indices, target=target_indices, value=sankey_df['count'])
    node = dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color=["blue"] * len(companies) + ["green"] * len(positions)
    )
    
    fig = go.Figure(data=[go.Sankey(node=node, link=link)])
    fig.update_layout(title_text="🔄 Flow from Companies to Positions", font_size=10, template="plotly_dark")
    return fig

# Updated plot_sankey function using plotly.express (Option A)
def plot_sankey_express(df: pd.DataFrame) -> px.Figure:
    """
    Generates a Sankey diagram showing flow between companies and positions using plotly.express.
    
    Parameters:
    - df (pd.DataFrame): Cleaned connections DataFrame.
    
    Returns:
    - plotly.graph_objects.Figure: Plotly Sankey diagram.
    """
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
            color=["blue"] * len(companies) + ["green"] * len(positions)
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=sankey_df['count']
        ),
        title="🔄 Flow from Companies to Positions",
    )
    fig.update_layout(template="plotly_dark")
    return fig

# Helper function to add background image (optional)
def add_bg_from_local(image_file: str):
    """
    Adds a background image to the Streamlit app.
    
    Parameters:
    - image_file (str): Path to the image file.
    """
    try:
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
    except FileNotFoundError:
        st.warning(f"⚠️ Background image `{image_file}` not found.")

# Main function to run the Streamlit app
def main():
    """Main function to run the Streamlit app."""
    # Optional: Add a background image
    # add_bg_from_local('media/background.png')  # Uncomment and set the correct path if needed

    # App title and description
    st.markdown(
        """
        <h1 style='text-align: center; color: white;'>🔗 LinkedIn Network Visualizer</h1>
        <h3 style='text-align: center; color: white;'>Unlock Insights from Your LinkedIn Connections</h3>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("📥 Upload Your LinkedIn Data")
    st.sidebar.markdown("Upload a ZIP file containing `Connections.csv` and optionally `messages.csv`.")

    # File uploader
    usr_file = st.sidebar.file_uploader("📁 Upload ZIP File", type={"zip"})

    if usr_file is not None:
        # Load connections data
        df_connections = get_data(usr_file, data_type="connections")
        # Load messages data
        df_messages = get_data(usr_file, data_type="messages")

        if df_connections is None or df_connections.empty:
            st.error("🚫 Connections data is empty or invalid. Please check your `Connections.csv` file.")
            return

        # Clean connections data
        df_clean = clean_connections_df(df_connections)

        if df_clean.empty:
            st.error("🚫 Cleaned connections data is empty. Please ensure your data is correctly formatted.")
            return

        # Display raw connections data
        with st.expander("📄 Show Raw Connections Data"):
            st.dataframe(df_connections)

        # Clean messages data if available
        if df_messages is not None and not df_messages.empty:
            df_clean_messages = clean_messages_df(df_messages)
            with st.expander("💬 Show Raw Messages Data"):
                st.dataframe(df_messages)
        else:
            df_clean_messages = pd.DataFrame()

        # Data aggregation
        agg_df_company = agg_sum(df_clean, "company")
        agg_df_position = agg_sum(df_clean, "position")

        total_connections = len(df_clean)

        # Display metrics
        st.metric("🔢 Total Connections", total_connections)
        if not agg_df_company.empty:
            st.metric("🏢 Top Company", f"{agg_df_company.iloc[0]['company']} ({agg_df_company.iloc[0]['count']})")
        if not agg_df_position.empty:
            st.metric("💼 Top Position", f"{agg_df_position.iloc[0]['position']} ({agg_df_position.iloc[0]['count']})")

        # Visualization sliders and filters
        st.sidebar.header("🎛️ Visualization Settings")
        top_n = st.sidebar.slider("Select Top N", min_value=5, max_value=50, value=10, step=1)

        # Top Companies and Positions Bar Charts
        st.subheader("📊 Top Companies & Positions")
        col1, col2 = st.columns(2)
        with col1:
            if not agg_df_company.empty:
                fig_company = plot_bar(agg_df_company, top_n, title="Top Companies")
                st.plotly_chart(fig_company, use_container_width=True)
            else:
                st.write("No company data available.")

        with col2:
            if not agg_df_position.empty:
                fig_position = plot_bar(agg_df_position, top_n, title="Top Positions")
                st.plotly_chart(fig_position, use_container_width=True)
            else:
                st.write("No position data available.")

        # Timeline of Connections
        st.subheader("⏰ Timeline of Connections")
        fig_timeline = plot_timeline(df_clean)
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Word Cloud of Chats
        if not df_clean_messages.empty:
            st.subheader("☁️ Word Cloud of Chats")
            fig_wordcloud = plot_wordcloud(df_clean_messages)
            if fig_wordcloud:
                st.pyplot(fig_wordcloud)
        else:
            st.info("📬 No messages data available for word cloud.")

        # Network Graph of Connections
        st.subheader("🔗 Network Graph of Connections")
        network_graph = plot_network_graph(df_clean)
        network_graph_html = network_graph.generate_html()
        # Ensure width is numeric to avoid TypeError
        st.components.v1.html(network_graph_html, height=750, width=1000)

        # Sankey Diagram: Companies to Positions
        st.subheader("🔄 Sankey Diagram: Companies to Positions")
        try:
            fig_sankey = plot_sankey_express(df_clean)
            st.plotly_chart(fig_sankey, use_container_width=True)
        except AttributeError:
            st.warning("⚠️ `plotly.express.sankey` is not available in your Plotly version. Using `plotly.graph_objects` instead.")
            fig_sankey = plot_sankey_go(df_clean)
            st.plotly_chart(fig_sankey, use_container_width=True)

    # Run the app
    if __name__ == "__main__":
        main()
