from opensearchpy import Float, Field, Integer, Document, Keyword, Text, Date, Object, InnerDoc

TOPIC_DIMENSIONS = 384
knn_params = {
    "name": "hnsw",
    "space_type": "cosinesimil",
    "engine": "nmslib"
}
index = 'topics-index'

class TopicKeyword(InnerDoc):
    name = Keyword()
    score = Float()

class KNNVector(Field):
    name = "knn_vector"
    def __init__(self, dimension, method, **kwargs):
        super(KNNVector, self).__init__(dimension=dimension, method=method, **kwargs)

class Topic(Document):
    vector = KNNVector(TOPIC_DIMENSIONS, knn_params)
    similarity_threshold = Float()
    created_at = Date()
    index = Integer()
    keywords = Object(TopicKeyword)
    name = Text()

    class Index:
        name = index

    def save(self, ** kwargs):
        return super(Topic, self).save(** kwargs)
