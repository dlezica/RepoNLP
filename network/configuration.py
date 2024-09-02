import os

class Config():
    def __init__(self):
        self.port = 9200
        self.host = os.getenv('OPENSEARCH_HOST', "localhost")
