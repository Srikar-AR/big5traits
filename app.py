from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import warnings
from pydantic import BaseModel
from mangum import Mangum
import torch
from transformers import BertForSequenceClassification,BertTokenizer
#import requests
warnings.filterwarnings("ignore")



app = FastAPI()
handler = Mangum(app)

#### 1. Model for Big 5 personality traits
model = BertForSequenceClassification.from_pretrained("/usr/share/Big-Five-Personality-Traits-Detection/Personality_detection_Classification_Save/", num_labels=5)#=num_labels)
tokenizer = BertTokenizer.from_pretrained('/usr/share/Big-Five-Personality-Traits-Detection/Personality_detection_Classification_Save/', do_lower_case=True) 
model.config.label2id= {
"Extroversion": 0,
"Neuroticism": 1,
"Agreeableness": 2,
"Conscientiousness": 3,
"Openness": 4,
}

model.config.id2label={
    "0": "Extroversion",
    "1": "Neuroticism",
    "2": "Agreeableness",
    "3": "Conscientiousness",
    "4": "Openness",}

def Personality_Detection_from_reviews_submitted (model_input: str) -> Dict[str, float]:
    if len(model_input)<20:
        return {"The sentence length is too small to predict the output"}
    else:
        dict_custom={}
        Preprocess_part1=model_input[:len(model_input)]
        Preprocess_part2=model_input[len(model_input):]
        dict1=tokenizer.encode_plus(Preprocess_part1,max_length=1024,padding=True,truncation=True)
        dict2=tokenizer.encode_plus(Preprocess_part2,max_length=1024,padding=True,truncation=True)
        dict_custom['input_ids']=[dict1['input_ids'],dict1['input_ids']]
        dict_custom['token_type_ids']=[dict1['token_type_ids'],dict1['token_type_ids']]
        dict_custom['attention_mask']=[dict1['attention_mask'],dict1['attention_mask']]
        outs = model(torch.tensor(dict_custom['input_ids']), token_type_ids=None, attention_mask=torch.tensor(dict_custom['attention_mask']))
        b_logit_pred = outs[0]
        pred_label = torch.sigmoid(b_logit_pred)
        ret ={
            "Extroversion": round(float(pred_label[0][0])*100, 2),
            "Neuroticism": round(float(pred_label[0][1])*100, 2),
            "Agreeableness": round(float(pred_label[0][2])*100, 2),
            "Conscientiousness": round(float(pred_label[0][3])*100, 2),
            "Openness": round(float(pred_label[0][4])*100, 2),}
        return ret 
    
class Interview(BaseModel):
    text_data: str

@app.post("/big5traits")
async def analyze_data(interview:Interview):
    try:
        text_data = interview.text_data
        traits = Personality_Detection_from_reviews_submitted(text_data)
        result = {"Big 5 personality traits ": traits}

        return JSONResponse(content=result)
    except Exception as e:
        print(e)
        # Handle exceptions and return an appropriate error response
        return HTTPException(status_code=500, detail=f"Error processing data: {str(e)}") 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload = True)