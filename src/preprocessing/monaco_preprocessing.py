from conllu_parser import ConlluParser
from tqdm import tqdm
import csv
import os

test_document_ids = ['Wieland', 'Seghers']

validate_document_ids = ['Gellert', 'Fontane']

def write_to_csv(documents, csv_dir: str):
    with open(csv_dir, 'w+', encoding='utf8', newline='') as f:  
        writer = csv.writer(f, delimiter='|')
        writer.writerow(['dataset', 'document_id', 'sent_id', 'clause_id', 'clause', 'contextualized_clause',  'gi', 'comment', 'nfr', 'generalization'])
        for document in tqdm(documents):
            if document.id in test_document_ids:
                dataset = 'test'
            elif document.id in validate_document_ids:
                dataset = 'validate'
            else:
                dataset = 'train'
            for passage in document.passages:
                for clause_number in passage.clauses.keys():
                    writer.writerow([
                                dataset, document.id, 
                                passage.sent_id, 
                                clause_number,
                                passage.clauses[clause_number].text, 
                                passage.clauses[clause_number].contextualized_text, 
                                passage.clauses[clause_number].gi,
                                passage.clauses[clause_number].comment,
                                passage.clauses[clause_number].nfr,
                                # invert the none-label
                                1 if int(passage.clauses[clause_number].gi[0]) == 0 else 0
                                ])
                    
def main(input_path=None, output_path=None):
    if input_path is None:
        input_path = os.path.join('..','..','data','korpus-public')
    if output_path is None:
        output_path = os.path.join('..','..','data','korpus-public.csv')
    documents = ConlluParser().create_documents(input_path)
    write_to_csv(documents, os.path.join(output_path))

if __name__ == '__main__':
    main()