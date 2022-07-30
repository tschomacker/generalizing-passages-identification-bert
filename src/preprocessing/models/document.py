from models.passage import Passage

class Document:

    def __init__(self, document_id, passages):
        self.id = document_id
        self.passages = passages
    
    def __getitem__(self, item):
        return self.passages[item]