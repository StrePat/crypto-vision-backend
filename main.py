from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
from prophet import Prophet

app = FastAPI()

# Autoriser les requêtes CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ajout de la route racine pour vérifier le déploiement
@app.get("/")
def read_root():
    return {"message": "Backend en ligne !"}

@app.get("/predict")
def predict():
    """
    Récupère les données historiques d'Ethereum sur 30 jours depuis CoinGecko,
    entraîne un modèle Prophet et prédit les prix pour les 3 prochains jours.
    """
    url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart?vs_currency=usd&days=30"
    response = requests.get(url)
    data = response.json()
    
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["ds"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["y"] = df["price"]
    df = df[["ds", "y"]]
    
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    
    future = model.make_future_dataframe(periods=3)
    forecast = model.predict(future)
    
    forecast_data = forecast[["ds", "yhat"]].tail(3)
    result = forecast_data.to_dict(orient="records")
    
    return result

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
