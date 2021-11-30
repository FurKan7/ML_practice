import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter

app = FastAPI()
router = InferringRouter()

@cbv(router)
class RunModel():
    def __init__(self):
        pkl_filename = "boston_model.pkl"
        # Load from file
        with open(pkl_filename, 'rb') as file:
            self.model = pickle.load(file)

    @router.get('/')
    def index(self):
        return {'message': 'Boston'}

    @router.get('/predict')
    def predict(self, crim: float, zn:float, indus:float, chas:float):
        x = np.array([[crim, zn, indus, chas]]) # shape (1, 4) 
        print(x.shape)
        res = self.model.predict(x)
        return {'result': f'{res.item():.4f}'}

app.include_router(router)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)