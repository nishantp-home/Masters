import pickle
import numpy as np
import os
import json
import tornado.ioloop
import tornado.web


if not os.path.exists('mymodel.pkl'):
    exit("Can't run without model !")

with open('mymodel.pkl') as f:
    model = pickle.load(f)

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, Tornado")


class PredictHandler(tornado.web.RequestHandler):
    def post(self):
        params = self.request.arguments
        x = np.array(map(float, params['input']))
        y = model.predict([x])[0]
        self.write(json.dumps({'prediction': y}))
        self.finish()


if __name__ == '__main__':
    application = tornado.web.Application([
        (r"/", MainHandler),
        (r"/predict", PredictHandler),
    ])

    application.listen(8888)
    tornado.ioloop.IOLoop.current().start()


