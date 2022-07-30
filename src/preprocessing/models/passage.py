from models.clause import Clause


class Passage:
    def __init__(self, sent_id, char_pos, text, clauses_dict, previous_passage, subsequent_passage):
        self.sent_id = sent_id
        self.char_pos = char_pos
        self.text = text
        self.clauses = clauses_dict
    
    def __repr__(self):
        return self.text
        
        