from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
import pandas as pd
from prophet import Prophet

app = FastAPI()

# Autoriser les requêtes CORS (on autorise toutes les origines pour simplifier)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route racine pour tester le déploiement
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
    
    # Construction du DataFrame pour Prophet
    df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    df["ds"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["y"] = df["price"]
    df = df[["ds", "y"]]
    
    # Instanciation et entraînement du modèle Prophet
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    
    # Création d'un DataFrame futur pour les 3 prochains jours
    future = model.make_future_dataframe(periods=3)
    forecast = model.predict(future)
    
    # Extraire les prévisions des 3 prochains jours
    forecast_data = forecast[["ds", "yhat"]].tail(3)
    result = forecast_data.to_dict(orient="records")
    
    return result

if __name__ == '__main__':
    import uvicorn
    # Utilise la variable d'environnement PORT, par défaut 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
