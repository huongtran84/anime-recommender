import streamlit as st
from pipeline.pipeline import AnimeRecommenderPipeline
from dotenv import load_dotenv


st.set_page_config(page_title="Anime Recommender", layout="wide")

load_dotenv()

@st.cache_resource
def get_pipeline():
    return AnimeRecommenderPipeline()

pipeline = get_pipeline()
st.title("Anime Recommender System")
query = st.text_input("Enter an anime title or description:")
if query:
    with st.spinner("Getting recommendations..."):
        try:
            recommendations = pipeline.get_recommendations(query)
            st.markdown("### Recommended Anime:")
            st.write(recommendations)
        except Exception as e:
            st.error(f"Error: {e}")