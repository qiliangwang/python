from flask import Flask
import flask_restful as restful
import os
import pickle


def load_data(file_name='res_data.pkl'):
    file_dir = os.path.join('./data', file_name)
    return pickle.load(open(file_dir, 'rb'))


app = Flask(__name__)
api = restful.Api(app)
data = load_data()


class SQL(restful.Resource):
    def get(self):
        return {'error': 0,
                'data': data,
                }


api.add_resource(SQL, '/api/sql')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
