from fastapi import FastAPI
import gradio as gr
from scipy.special import softmax
from transformers import AutoTokenizer,AutoConfig,AutoModelForSequenceClassification
import numpy as np
from pydantic import BaseModel

# Setup
model_path = f"bambadij/Tweet_sentiment_analysis_Distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model =AutoModelForSequenceClassification.from_pretrained(model_path)


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Input preprocessing
text = "Covid cases are increasing fast!"
text = preprocess(text)

# PyTorch-based models
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
scores = output[0][0].detach().numpy()
scores = softmax(scores)

def sentiment_analysis(text):
    text =preprocess(text)
    #Pytorch-based models
    encoded_input = tokenizer(text,return_tensors='pt')
    output = model(**encoded_input)
    scores_ = output[0][0].detach().numpy()
    scores_ =softmax(scores_)

    #Foramt ouptput dict of scores
    labels =['Negative','Neutral','Positive']
    scores = {l:float(s) for (l,s) in zip(labels,scores_)}
    return scores

#INPUT MODELING
class ModelInput(BaseModel):

    tweet:str
    
app = FastAPI()

@app.get("/")
async def root():
    return {"message : Hello la team"}

@app.post("/tweets")
async def run(input : ModelInput):
    result = sentiment_analysis(text=input.tweet)
    return {
           "input_text":input.tweet,
           "confidence_scores" :result  
            }
    