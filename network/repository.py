
from network.database import Database

class Repository():
    def __init__(self):
        self.client = Database()
    
    def get_client(self):
        return self.client.get_client()