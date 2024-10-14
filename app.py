# Import libraries
import re
import streamlit as st
import pandas as pd
from zipfile import ZipFile
import shutil
from rapidfuzz import process, fuzz
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

# Helper functions
def get_data(usr_file, data="connections") -> pd.DataFrame:
    """Extracts CSV data from a zip file."""
    if usr_file is None:
        return None

    with ZipFile(usr_file, "r") as zipObj:
        zipObj.extractall("/tmp")  # Extract to /tmp for Streamlit Cloud

    if data == "connections":
        raw_df = pd.read_csv("/tmp/Connections.csv", skiprows=3)
    elif data == "messages":
        raw_df = pd.read_csv("/tmp/messages.csv")
    else:
        return None

    # Delete the extracted data folder
    shutil.rmtree("/tmp", ignore_errors=True)

    return raw_df


def clean_df(df: pd.DataFrame, privacy: bool = False) -> pd.DataFrame:
    """Cleans the dataframe containing LinkedIn connections data."""
    if privacy:
        df.drop(columns=["first_name", "last_name", "email_address"], inplace=True)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Drop missing values in company and position
    df.dropna(subset=["company", "position"], inplace=True)

    # Combine first name and last name
    if 'first_name' in df.columns and 'last_name' in df.columns:
        df['name'] = df['first_name'] + ' ' + df['last_name']
        df.drop(columns=["first_name", "last_name"], inplace=True)

    # Truncate company names
    if 'company' in df.columns:
        df['company'] = df['company'].str[:35]

    # Convert 'connected_on' to datetime if column exists
    if 'connected_on' in df.columns:
        df['connected_on'] = pd.to_datetime(df['connected_on'], errors='coerce')
        df.dropna(subset=['connected_on'], inplace=True)

    # Filter out unwanted companies
    if 'company' in df.columns:
        df = df[~df['company'].str.contains(r"[Ff]reelance|[Ss]elf-[Ee]mployed|\\.|-", regex=True)]

    # Fuzzy match for positions
    if 'position' in df.columns:
        replace_fuzzy_match(df, "position", "Data Scientist")
        replace_fuzzy_match(df, "position", "Software Engineer", min_ratio=85)

    return df


def replace_fuzzy_match(df: pd.DataFrame, column: str, query: str, min_ratio: int = 75):
    """Replaces fuzzy matches in the specified column with the query string."""
    if column in df.columns:
        pos_names = df[column].unique()
        matches = process.extract(query, pos_names, scorer=fuzz.ratio, limit=500)
        matching_pos_name = [match[0] for match in matches if match[1] >= min_ratio]
        matches_rows = df[column].isin(matching_pos_name)
        df.loc[matches_rows, column] = query


def agg_sum(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Aggregates value counts for companies and positions."""
    if name in df.columns:
        df = df[name].value_counts().reset_index()
        df.columns = [name, "count"]
        df = df.sort_values(by="count", ascending=False)
        return df
    return pd.DataFrame(columns=[name, "count"])


def plot_bar(df: pd.DataFrame, rows: int, title=""):
    """Creates a bar plot for the top N entries in the dataframe."""
    if df.empty:
        st.warning(f"No data available for {title}")
        return None
    
    height = 500 if rows <= 25 else 900
    fig = px.bar(
        df.head(rows),
        x='count',
        y='company' if 'company' in df.columns else 'position',
        template="plotly_dark",
        hover_data={df.columns[1]: False},
    )
    fig.update_layout(
        height=height,
        width=600,
        margin=dict(pad=5),
        hovermode="y",
        yaxis_title="",
        xaxis_title="",
        title=title,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_timeline(df: pd.DataFrame):
    """Generates a timeline plot of connections over time."""
    if 'connected_on' not in df.columns:
        st.warning("No 'connected_on' column found in the data.")
        return None
    
    df = df["connected_on"].value_counts().reset_index()
    df.rename(columns={"index": "connected_on", "connected_on": "count"}, inplace=True)
    df = df.sort_values(by="connected_on", ascending=True)
    fig = px.line(df, x="connected_on", y="count")
    fig.update_layout(
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
        xaxis_title="Date",
    )
    return fig


def plot_wordcloud(chats: pd.DataFrame):
    """Generates a word cloud from chat messages."""
    if 'subject' not in chats.columns or 'content' not in chats.columns:
        st.warning("Required columns for wordcloud generation are missing.")
        return None
    
    chats_nospam = chats[chats['subject'].isnull()]
    chats_nospam_nohtml = chats_nospam[~chats_nospam['content'].str.contains("<|>")]
    messages = chats_nospam_nohtml.dropna(subset=["content"])['content'].values
    corpus = []

    for message in messages:
        message = re.sub(r"http[s]?\S+", "", message)  # Remove URLs
        message = re.sub("[^a-zA-Z]", " ", message).lower()  # Keep only letters
        message = re.sub(r"\s+[a-zA-Z]\s+", " ", message)  # Remove singular letters
        words = [word for word in message.split() if word not in set(stopwords.words("english"))]
        corpus.append(" ".join(words))

    linkedin_mask = np.array(Image.open("media/linkedin.png"))
    wordcloud = WordCloud(
        width=3000,
        height=3000,
        background_color="black",
        stopwords=STOPWORDS,
        mask=linkedin_mask,
        contour_color="white",
        contour_width=2,
    ).generate(" ".join(corpus))

    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return plt


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Linkedin Network Visualizer", page_icon="🕸️", layout="wide")
    
    st.markdown(
        """
        <h1 style='text-align: center; color: white;'>Linkedin Network Visualizer</h1>
        <h3 style='text-align: center; color: white;'>The missing feature in LinkedIn</h3>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("First, upload your data 💾")
    st.caption("Don't know where to find it? [Click here](https://github.com/benthecoder/linkedin-visualizer/tree/main/data_guide#how-to-get-the-data).")
    
    # Upload files
    usr_file = st.file_uploader("Drop your zip file 👇", type={"zip"})
    df_ori = get_data(usr_file)

    if df_ori is None:
        st.warning("Please upload a zip file containing the data.")
        return

    df_clean = clean_df(df_ori)

    with st.expander("Show raw data"):
        st.dataframe(df_ori)

    # Data wrangling
    agg_df_company = agg_sum(df_clean, "company")
    agg_df_position = agg_sum(df_clean, "position")
    
    total_conn = len(df_ori)
    
    # Display metrics
    st.metric("Total Connections", total_conn)
    
    # Visualizations
    st.subheader("Top Companies & Positions")
    top_n = st.slider("Select Top N", 1, 50, 10)
    
    company_plt = plot_bar(agg_df_company, top_n, title="Top Companies")
    if company_plt:
        st.plotly_chart(company_plt, use_container_width=True)
    
    position_plt = plot_bar(agg_df_position, top_n, title="Top Positions")
    if position_plt:
        st.plotly_chart(position_plt, use_container_width=True)

    st.subheader("Timeline of Connections")
    timeline_plt = plot_timeline(df_clean)
    if timeline_plt:
        st.plotly_chart(timeline_plt, use_container_width=True)

    st.subheader("Wordcloud of Chats")
    chats = get_data(usr_file, data="messages")
    wordcloud_plt = plot_wordcloud(chats)
    if wordcloud_plt:
        st.pyplot(wordcloud_plt)


if __name__ == "__main__":
    main()
