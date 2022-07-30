class LimeClause():
    """
    Data Object that contains all necessary information to apply Lime to a clause
    """
    def __init__(self, korpus_df, document_id, sent_id, clause_id):
        self.document_id = document_id
        self.sent_id = sent_id
        self.clause_id = clause_id
        filtered_df = korpus_df[(korpus_df['document_id']  == self.document_id) 
                              & (korpus_df['sent_id'] == self.sent_id) 
                              & (korpus_df['clause_id'] == self.clause_id)]
        self.clause = filtered_df['clause'].iloc[0]
        self.contextualized_clause = filtered_df['contextualized_clause'].iloc[0]
        self.gi =  filtered_df['gi'].iloc[0]
        self.comment = filtered_df['comment'].iloc[0]
        self.nfr =  filtered_df['nfr'].iloc[0]
        self.file_name = self.document_id+'_'+self.sent_id+'_'+self.clause_id
    
    #def __str__():
    #    self.clause