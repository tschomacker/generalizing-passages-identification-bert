import pandas
from tqdm import tqdm

POSSIBLE_GI_SUBTAGS = ['ALL','BARE','DIV','EXIST','MEIST','NEG']
POSSIBLE_GS_SUBTAGS =['GENERIC_SENTENCE','GENERALIZING_SENTENCE']


def create_sitent_df(sitent_text_path, sitent_train_test_path, verbose = True):
    """
    create a df from the sitent corpus
        gs_tag :
        gi_tags :
        gi_tags_vector : 
    """
    sitent_df = pandas.read_json(path_or_buf=sitent_text_path, dtype = str)
    sitent_df['new_sent'] = sitent_df['new_sent'].map({'True': True, 'False': False})
    sitent_df["clause"] = sitent_df["text"]
    
    sitent_df["gi_tags_vector"] = sitent_df["gi_tags"].apply(lambda x: _subtags_to_vector(x,POSSIBLE_GI_SUBTAGS))
    sitent_df["gs_tag_vector"] = sitent_df["gs_tag"].apply(lambda x: _subtags_to_vector(x,POSSIBLE_GS_SUBTAGS))
    sitent_df["document_id"] = sitent_df["instanceid"].apply(lambda x: _instanceid_to_doc_id(x))
    train_test_df = pandas.read_csv(sitent_train_test_path, sep='\t', dtype = str)
    sitent_df["category"] = sitent_df["document_id"].apply(lambda x: _doc_id_to_category(x, train_test_df)) 
    sitent_df["dataset"] = sitent_df["document_id"].apply(lambda x: _doc_id_to_train_test(x, train_test_df)) 
    sitent_df["sent_id"] = _calculate_sent_id(sitent_df)
    sitent_df['clause_id'] =  _calculate_clause_id(sitent_df, verbose)
    sitent_df["contextualized_clause"] = _contextualize_clauses(sitent_df, verbose)
    return sitent_df

def _contextualize_clauses(sitent_df, verbose = True):
    """
    Note: depends on the correct calculation of clause_id, document_id and sent_id
    return:
        pandas.Series will the contextualized representation of all clauses
    """
    contextualized_clause_list = []
    for _index, row in tqdm(sitent_df.iterrows(), total=len(sitent_df), desc='clause(s) contextualized', disable=not verbose):
        # contextualiz in its own sentence
        all_clauses_in_same_document = sitent_df[(sitent_df.document_id == row.document_id)]
        
        all_clauses_in_same_sentence = all_clauses_in_same_document[(all_clauses_in_same_document.sent_id == row.sent_id)]
        # contextualize in its own sentence
        contextualized_in_sentence = ''
        for i in range(1,len(all_clauses_in_same_sentence)+1):
            
            current_clause_df = all_clauses_in_same_sentence[(all_clauses_in_same_sentence.clause_id == i)]
            current_clause_df.reset_index(drop=True)
            current_clause = current_clause_df.iloc[0]
            if current_clause.clause_id == row.clause_id:
                current_clause_text = ' <b> '+current_clause.clause+' </b>'
            else:
                current_clause_text = ' '+current_clause.clause
            contextualized_in_sentence += current_clause_text
            
        current_sent_id = row.sent_id
        # get predecessor 
        ## first sentence has no predecessor -> skip
        predecessor = ''
        if 1 != row.sent_id:
            # build predecessor
            predecessor_clauses_df = all_clauses_in_same_document[(all_clauses_in_same_document.sent_id == current_sent_id-1)]
            
            for _index, row in predecessor_clauses_df.iterrows():
                predecessor = predecessor + ' ' + row.clause
        # get successor
        successor = ''
        successor_clauses_df = all_clauses_in_same_document[(all_clauses_in_same_document.sent_id == current_sent_id+1)]

        ## last sentence has no successor -> skip
        try: 
            for _index, row in successor_clauses_df.iterrows():
                successor = successor + ' ' + row.clause
        except:
            print('is the last sentence')
            
        contextualized_clause = predecessor+contextualized_in_sentence+successor
        contextualized_clause_list.append(contextualized_clause)    
    return contextualized_clause_list

def _calculate_sent_id(sitent_df):
    """
    Note: depends on the correct calculation of document_id
    return:
        pandas.Series will all sent_ids
    """
    sent_id_count = 1
    current_document_id = ''
    #for sample in sitent_df:
    sent_id_list = []
    for _index, row in sitent_df.iterrows():
        # new document has started 
        if current_document_id != row.document_id:
            # reset the sent_id count
            sent_id_count = 1
            current_document_id = row.document_id
            start_of_document = True
        else:
            start_of_document = False
        
        # increase the count when a new sentence starts
        # except when the start of the sentence is the start of a new document
        if row.new_sent & (not start_of_document):
            sent_id_count += 1
        
        # assign sent_id
        sent_id_list.append(sent_id_count)
    return pandas.Series(sent_id_list) 

def _doc_id_to_train_test(doc_id,categories_df):
    validation_corpus = 'MONACO'
    if 'MONACO' == validation_corpus:
        return 'train'
    elif 'CAGE' == validation_corpus:
        series_category = categories_df[(categories_df.category_filename == doc_id)]
        if 1 == len(series_category['fold']):
            sitent_fold = series_category['fold'].iloc[0]
            if 'test' == sitent_fold:
                return 'validate'
            elif 'train' == sitent_fold:
                return 'train'
            else:
                raise Exception('unable to map:',doc_id)
        else:
            raise Exception('unable to map:',doc_id)

def _doc_id_to_category(doc_id,categories_df):
    series_category = categories_df[(categories_df.category_filename == doc_id)]
    if 1 == len(series_category['category']):
        return series_category['category'].iloc[0]
    else:
        raise Exception('unable to map:',doc_id)

def _instanceid_to_doc_id(instanceid):
    return instanceid.rsplit('_', 1)[0]

def _calculate_clause_id(sitent_df,verbose):
    """
    Note: depends on the correct calculation of sent_id
    return:
        pandas.Series will all clause_id
    """
    clause_id_count = 1
    clause_id_list = []
    for _index, row in tqdm(sitent_df.iterrows(), disable=not verbose, desc='clause id(s) calculated'):
        if row.new_sent:
            clause_id_count = 1
        else:
            clause_id_count += 1
        clause_id_list.append(clause_id_count)
    return pandas.Series(clause_id_list) 

def _subtags_to_vector(subtags_str, possible_subtags):
    """
    return: multi-hot vector for the subtags of the tag in string format
    based on naacl22/mona/models/clause
    """
    label = ''
    #used for labeling entries without any label
    has_label = False
    
    subtags = subtags_str.split(',')
    if 'None' == subtags[0]:
        subtags = []
    

    for possible_subtag in possible_subtags:
        if possible_subtag in subtags:
            label += '1'
            has_label = True
        else:
            label += '0'
    if has_label:
        label = '0'+label
    else:
        label = '1'+label
        if len(subtags) > 0:
            _tag_vector_sanity_check(subtags,possible_subtags)
    return label


def _tag_vector_sanity_check(subtags,possible_subtags):
    """
    check for tags, that are unknown or not mapped properly
    """
    for assigned_subtag in subtags:
        found_it_in = ''
        for possible_subtag in possible_subtags:
            if assigned_subtag == possible_subtag:
                found_it_in = possible_tag
    if found_it_in == '':
        raise Exception('unable to map:',assigned_subtag,'from\n')
        
        
