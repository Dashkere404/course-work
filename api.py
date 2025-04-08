from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from scripts.recommendation import recommend_by_tags
from scripts.movie_clustering import perform_clustering

app = FastAPI()

# Загрузка данных при запуске сервера
movies_df = pd.read_csv("output/clusters_movies_with_tags.csv")

class MovieRecommendationRequest(BaseModel):
    movie_id: int

@app.post("/recommend/")
def recommend_movie(request: MovieRecommendationRequest):
    """
    Рекомендация фильмов на основе ID фильма.
    """
    movie_id = request.movie_id
    if movie_id not in movies_df['movieId'].values:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    recommended_movies = recommend_by_tags(movie_id, movies_df)
    return {"recommended_movies": recommended_movies}

@app.get("/clusters/")
def get_clusters():
    """
    Получение информации о кластерах фильмов.
    """
    clusters = movies_df.groupby("cluster").size().to_dict()
    return {"clusters": clusters}
