from flask import Flask
from flask_restx import Api, Namespace, Resource, reqparse, fields
import werkzeug.exceptions as HTTPException
from flask_cors import CORS

app = Flask(__name__)

api = Api(
    app=app,
    version='1.0',
    title='캡스톤 모델 API'
    )

CORS(app)

modelNs = Namespace('model')
api.add_namespace(modelNs)


parser = api.parser()
parser.add_argument('model_name', type=str, help='One of [tranformer, linear, lstm]')
parser.add_argument('model_pth', type=str, help='Exact path of pth file')
parser.add_argument('date', type=str, help='Type of date string like YYYY-MM-DD')
@modelNs.route('/callModel')
class callModel(Resource):
    @modelNs.expect(parser)
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('model_name', type=int, default=1)
        parser.add_argument('model_pth')
        parser.add_argument('date')
        
        allow_models = {'transformer', 'linear', 'lstm'}
        return {'price': 12345}
        
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8888)