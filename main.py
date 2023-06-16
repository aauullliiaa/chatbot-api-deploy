import os
import uvicorn
import traceback
import tensorflow as tf

import re
import random
import numpy as np

from pydantic import BaseModel
from fastapi import FastAPI, Response
from keras.utils import pad_sequences
from utils import tokenizer, lbl_enc, df, X

model = tf.keras.models.load_model('./model.h5')

app = FastAPI()

# This endpoint is for a test (or health check) to this server
@app.get("/")
def index():
    return "Hello world from ML endpoint!"

class RequestText(BaseModel):
    text:str

@app.post("/predict_text")
def predict_text(req: RequestText, response: Response):
    try:
        req_text = req.text
        print("Uploaded text:", req_text)
        
        # Step 1: Text preprocessing
        text = []
        txt = re.sub('[^a-zA-Z\']', ' ', req_text)
        txt = txt.lower()
        txt = txt.split()
        txt = " ".join(txt)
        text.append(txt)
        
        # Step 2: Prepare data to model
        x_test = tokenizer.texts_to_sequences(text)
        x_test = np.ravel(x_test)
        x_test = pad_sequences([x_test], padding='post', maxlen=X.shape[1])
        
        # Step 3: Predict the data
        y_pred = model.predict(x_test, verbose=0)
        y_pred = y_pred.argmax()
        tag = lbl_enc.inverse_transform([y_pred])[0]
        
        # Step 4: Change the result of determined API output
        responses = df[df['tag'] == tag]['responses'].values[0]
        next_patterns = df[df['tag'] == tag]['next-patterns'].values[0]

        return {
            "you" : req_text,
            "bot" : random.choice(responses),
            "next_patterns" : next_patterns
        }
    
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"


# Starting the server
port = os.environ.get("PORT", 4000)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0',port=port)