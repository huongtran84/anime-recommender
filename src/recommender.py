from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from src.prompt_template import get_anime_prompt

class AnimeRecommender:
    def __init__(self, retriever, api_key:str, model_name:str):
        self.model = ChatGroq(api_key=api_key, model=model_name)
        prompt_template = get_anime_prompt()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )

    def get_recommendations(self, query: str):
        result = self.qa_chain({"query": query})
        return result["result"]