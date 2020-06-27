from GRAMSMOT import GRAMSMOT

class Models(object):
    def __init__(self,params):
        self.model     = GRAMSMOT(params)

    def get_model(self):
        return self.model
