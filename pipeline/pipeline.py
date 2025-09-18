from src.vector_store import VectorStoreBuilder
from src.recommender import AnimeRecommender
from config.config import MODEL_NAME, GROQ_API_KEY
from utils.logger import get_logger
from utils.custom_exception import CustomException

logger = get_logger(__name__)

class AnimeRecommenderPipeline:
    def __init__(self,persist_dir:str = "chroma_db"):
        try:
            logger.info("Initializing VectorStoreBuilder...")
            vector_builder = VectorStoreBuilder(csv_path="", persist_dir=persist_dir)
            retriever = vector_builder.load_vector_store().as_retriever()
            self.recommender = AnimeRecommender(
                retriever=retriever,
                api_key=GROQ_API_KEY,
                model_name=MODEL_NAME
            )
            logger.info("AnimeRecommenderPipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing AnimeRecommenderPipeline: {e}")
            raise CustomException(e)

    def get_recommendations(self, query: str) -> str:
        try:
            logger.info(f"Getting recommendations for query: {query}") 
            recommendations = self.recommender.get_recommendations(query)
            logger.info("Recommendations retrieved successfully.")
            return recommendations
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            raise CustomException(e)