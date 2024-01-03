from fastapi import FastAPI
from transformers import pipeline,BertForSequenceClassification,BertTokenizerFast
from pydantic import BaseModel
import joblib
app = FastAPI()



nlp = joblib.load("/Users/macmini/Desktop/bert_fast_api/nlp-model.ipynb")


class Input(BaseModel):
    text : str





@app.post("/")
async def root(input : Input):
    print(nlp(input.text))
    return nlp(input.text)
