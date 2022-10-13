from conllu_parser import ConlluParser
from tqdm import tqdm
import csv
import os
import argparse
import re
import pandas
import warnings

def write_to_csv(documents, csv_dir: str, test_document_ids, validate_document_ids, exclude_documents=[]):
    with open(csv_dir, 'w+', encoding='utf8', newline='') as f:  
        writer = csv.writer(f, delimiter='|')
        writer.writerow(['dataset', 'document_id', 'sent_id', 'clause_id', 'clause', 'contextualized_clause',  
                         'gi','gi_none', 'comment','comment_none', 'nfr','nfr_none', 'nfr_ex_mk', 'generalization', 
                         'reflexive_ex_mk_binary', 'reflexive_ex_mk_multi'])
        for document in tqdm(documents):
            if document.id not in exclude_documents:
                if document.id in test_document_ids:
                    dataset = 'test'
                elif document.id in validate_document_ids:
                    dataset = 'validate'
                else:
                    dataset = 'train'

                for passage in document.passages:
                    for clause_number in passage.clauses.keys():
                        writer.writerow([
                                    dataset, 
                                    document.id, 
                                    passage.sent_id, 
                                    clause_number,
                                    passage.clauses[clause_number].text, 
                                    passage.clauses[clause_number].contextualized_text, 
                                    passage.clauses[clause_number].gi,
                                    passage.clauses[clause_number].gi_none,
                                    passage.clauses[clause_number].comment,
                                    passage.clauses[clause_number].comment_none,
                                    passage.clauses[clause_number].nfr,
                                    passage.clauses[clause_number].nfr_none,
                                    passage.clauses[clause_number].nfr_ex_mk,
                                    passage.clauses[clause_number].generalization,
                                    passage.clauses[clause_number].reflexive_ex_mk_binary,
                                    passage.clauses[clause_number].reflexive_ex_mk_multi

                                    ])
    sanity_check(csv_dir, test_document_ids, validate_document_ids)
    

def sanity_check(csv_dir, test_document_ids, validate_document_ids):
    korpus_df = pandas.read_csv(csv_dir, dtype=str, delimiter='|')
    for test_doc in test_document_ids+validate_document_ids:
        if len(korpus_df[(korpus_df['document_id'] == test_doc)]) < 1:
            warnings.warn(test_doc, 'was not found as a doc id')
    copyright_protected_text_ids = ['Mann', 'Seghers']
    for text_id in copyright_protected_text_ids:
        # reset index to later use the same low index number
        s = korpus_df[(korpus_df['document_id'] == text_id)].reset_index()
        if re.match('^[ -]*$', s.clause[42]):
            warnings.warn(text_id, 'has only placeholder content') 
    
                    
def main(input_path, output_path, test_document_ids, validate_document_ids,exclude_documents):
    if input_path is None:
        input_path = os.path.join('..','..','data','korpus-public')
    if output_path is None:
        output_path = os.path.join('..','..','data','korpus-public.csv')
    documents = ConlluParser().create_documents(input_path)
    write_to_csv(documents, os.path.join(output_path), test_document_ids, validate_document_ids,exclude_documents)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing MONACO by parsing conllu files and applying a train, validate, test split')
    parser.add_argument('--output', type=str, default='../../data/korpus-public.csv', help="Path of the output file.")
    parser.add_argument('--test', nargs='+', default=['Wieland', 'Seghers'], help="ID(s) of the document(s) in the TEST dataset")
    parser.add_argument('--validate', nargs='+', default=['Fontane','Mann'], help="ID(s) of the document(s) in the VALIDATE dataset")
    parser.add_argument('--exclude', nargs='+', default=[], help="ID(s) of the document(s) that should not be considered in the datasets at all.")
    args = parser.parse_args()
    
    main(None, args.output, args.test, args.validate, args.exclude)