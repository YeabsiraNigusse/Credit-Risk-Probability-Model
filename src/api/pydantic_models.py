from pydantic import BaseModel

class TransactionFeatures(BaseModel):
    TransactionStartTime: str   # e.g. "2025-07-26T19:00:00"
    AccountId:             str
    Amount:                float
    Value:                 float
    ProductCategory:       str
    CurrencyCode:          str
    CountryCode:           str
    ProviderId:            str
    ChannelId:             str
    PricingStrategy:       str


class PredictionResponse(BaseModel):
    risk_probability: float
