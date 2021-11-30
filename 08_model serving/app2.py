import uvicorn
from fastapi import FastAPI
from fastapi_utils.cbv import cbv
from fastapi_utils.inferring_router import InferringRouter

app = FastAPI()
router = InferringRouter()

@cbv(router)
class RunModel():
    @router.get('/')
    def index(self):
        return {'message': 'Hello'}

    @router.get('/predict')
    def get_res(self, feat1: float, feat2:float):
        res = feat1 + feat2
        return {'result': f'{res:.4f}'}

app.include_router(router)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)