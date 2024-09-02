from opensearchpy import connections, OpenSearch
from network.configuration import Config
from network.auth import credentials

class Database():
    def __init__(self, ):
        config = Config()
        self.client = self.create_connection(config.host, config.port, credentials)

    def create_connection(self, host, port, credentials) -> OpenSearch:
            return connections.create_connection(
            hosts = [{'host': host, 'port': port}],
            http_compress = True, 
            http_auth = credentials,
            use_ssl = False,
            verify_certs = False,
            ssl_assert_hostname = False,
            ssl_show_warn = False,
            alias='default'
        )

    def create_index(self, index):
        index_body = {    
            'settings': {        
                'index': {            
                    'number_of_shards': 1,
                    'knn': True      
                }    
            }
        }
        response = "already created"
        if not self.client.indices.exists(index=index):
            response = self.client.indices.create(index, body=index_body)
        
        return response
    
    def search(self, query, index, size):
        respose = self.client.search(
            body=query,
            index=index,
            size= size
        )

        return respose
    
    def get_client(self):
        return self.client
    
        