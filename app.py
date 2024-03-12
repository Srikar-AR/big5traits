from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import warnings
from pydantic import BaseModel
from mangum import Mangum
import torch
from transformers import BertForSequenceClassification,BertTokenizer
import os
import random
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


os.environ['TRANSFORMERS_CACHE'] = "/tmp/transformers_cache"


app = FastAPI()
handler = Mangum(app)

#### 1. Model for Big 5 personality traits
model = BertForSequenceClassification.from_pretrained("/usr/share/Personality_detection_Classification_Save/", num_labels=5)#=num_labels)
#model = BertForSequenceClassification.from_pretrained("./Big-Five-Personality-Traits-Detection/Personality_detection_Classification_Save/", num_labels=5)#=num_labels)
tokenizer = BertTokenizer.from_pretrained('/usr/share/Personality_detection_Classification_Save/', do_lower_case=True) 
#tokenizer = BertTokenizer.from_pretrained('./Big-Five-Personality-Traits-Detection/Personality_detection_Classification_Save/', do_lower_case=True) 

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

def find_most_similar_sentence(target_sentence, candidate_sentences):
    vectorizer = TfidfVectorizer()
    candidate_vectors = vectorizer.fit_transform(candidate_sentences)
    target_vector = vectorizer.transform([target_sentence])
    similarity_scores = cosine_similarity(target_vector, candidate_vectors)
    most_similar_index = similarity_scores.argmax()
    most_similar_sentence = candidate_sentences[most_similar_index]
    #similarity_score = similarity_scores[0, most_similar_index]

    return most_similar_sentence

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
def get_recommendations(traits, text):
    dict_traits = {}
    if traits["Extroversion"] <60:
        ext_list = ["be more outgoing and talkative", "thrive in social situations", 
                    "have a wide social circle and find it easy to make friends",
                    "like to start conversations feel comfortable arguing and debating your opinions",
                    "seek excitement","generally enjoy being around people", "work in a supervisor position with others"]
        dict_traits['Extroversion'] = find_most_similar_sentence(text, ext_list)

    if traits['Neuroticism'] <60:
        neur_list = ["Don't feel insecure", "Don't get stressed easily", "Don't be irritable or moody to others", 
                     "Need not to worry a lot", "Don't feel sad"]
        dict_traits["Neuroticism"] = find_most_similar_sentence(text, neur_list)

    if traits['Agreeableness'] <60:
        agre_list = [ "Try being kind to others", "Try being empathetic",  "Always helpful Others", "Try caring for others",  
                        "Be compassionate towards the work you do",  "Always be Trustworthy"]
        dict_traits["Agreeableness"] = find_most_similar_sentence(text, agre_list)

    if traits['Conscientiousness']<60:
        cons_list = ["Be more optimistic", " Try being emotionally stable",  "Not to unlikely react in a stressful environment",
                    "Be well-organised and hardworking",  "Be detailed-oriented in work", " Try to be good at planning", 
                    "mindful of deadlines", "Always be goal-driven"] 
        dict_traits["Conscientiousness"] = find_most_similar_sentence(text, cons_list)

    if traits['Openness'] <60:
        open_list = ["Try to enjoy learning and trying new things", "Always have an active imagination", "be more creative", 
                    "be intellectually curious", "think about abstract concepts","Try to enjoy new challenges", 
                    "Need to have a wide range of interests"]
        dict_traits["Openness"] = find_most_similar_sentence(text, open_list)
    
    dict_traits = {k:dict_traits[k] for k in random.sample(list(dict_traits.keys()), len(dict_traits))}
    return dict_traits

class Interview(BaseModel):
    text_data: str

@app.post("/big5traits")
async def analyze_data(interview:Interview):
    try:
        text_data = interview.text_data
        traits = Personality_Detection_from_reviews_submitted(text_data)
        recommendations = get_recommendations(traits, text_data)

        Area_of_Improvements = []
        Actionable_Recommendations = []
        for key, value in recommendations.items():
            Area_of_Improvements.append(key)
            Actionable_Recommendations.append(value)
        result = {"Big_5_personality_traits ": traits,
                  "Area_of_Improvements":Area_of_Improvements,
                  "Actionable_Recommendations": Actionable_Recommendations}

        return JSONResponse(content=result)
    except Exception as e:
        print(e)
        # Handle exceptions and return an appropriate error response
        return HTTPException(status_code=500, detail=f"Error processing data: {str(e)}") 

import os

def list_files_and_directories(root_directory):
    file_list = []
    directory_list = []

    for root, dirs, files in os.walk(root_directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
            print("File:", file_path)

        for directory in dirs:
            directory_path = os.path.join(root, directory)
            directory_list.append(directory_path)
            print("Directory:", directory_path)

@app.get("/testing")
async def testing():
    root_path = "/tmp"
    list_files_and_directories(root_path)
    return {"testing":"testing"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload = True)