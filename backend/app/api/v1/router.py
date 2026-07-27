from fastapi import APIRouter

from app.api.v1 import auth, players, predictions, rankings

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(players.router)
api_router.include_router(predictions.router)
api_router.include_router(rankings.router)
