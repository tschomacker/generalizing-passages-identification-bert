from io import open

import os
from conllu import parse
from models.document import Document
from models.clause import Clause
from models.passage import Passage
from tqdm import tqdm
import re


class ConlluParser:

    def __init__(self):
        self.fields = ['id', 'form', 'lemma', 'upos', 'xpos', 'feats', 'head',
            'deprel', 'deps', 'misc', 'gi', 'pgi', 'comment', 'pcomment', 'nfr', 'pnfr']
        # used for extracting the document_id from the file path
        # this has to changed depending on your os
        #self.document_id_separator = "/"

    def create_documents(self, rootdir):
        """
        Scans rootdirectory and its subdirectories for all conllu files and creates Documents, Passages and Clauses
        from the files
        return: list<Document>
        """
        documents = []
        i = 0
        for conllu_file in tqdm(self._get_all_conllu_files(rootdir), desc = 'Creating document(s) from conllu file(s)'):
            # start creating a document
            # 1. generate document id
            document_id = conllu_file.split('__')[0]
            document_id = re.split(r'[\\,\/]', document_id)[-1]
            # 2.a create all passages in the document
            document_data = open(conllu_file, "r", encoding="utf-8").read()
            passages_token_list = parse(document_data, fields=self.fields, field_parsers={
                'misc': self._misc_parser, "gi": self._field_parser, 'pgi': self._field_parser, 'comment': self._field_parser,
                'pcomment': self._field_parser, 'nfr': self._field_parser, 'pnfr': self._field_parser
            })
            # 2.b tranform tokenlist to passages
            passages = []
            previous_passage = None
            for passage_token_list_item in passages_token_list:
                # 3 create a passage
                # 3.a create identifiers
                sent_id = passage_token_list_item.metadata['sent_id']
                char_pos = passage_token_list_item.metadata['char_pos']
                # 3.b create passage text
                passage_text = self._create_passage_text(
                    passage_token_list_item)
                # 3.c add previous and subsequent passage
                previous_passage = None
                subsequent_passage = None

                # 4. create the clauses in the passage
                # clauses = self._extract_clauses(token_list)

                # 4.a create a dict with all clauses in the passage
                clauses_dict = {}
                clause_number = 1
                for token in passage_token_list_item:
                    #print(document_id,sent_id,clause_number)
                    if token.get('misc') is not None:
                    # punctuation has no clause number so I assign them
                    # to the previous clause. So the clause_number
                    # remains unchanged in this case
                        clause_number = token.get('misc')

                    if clause_number not in clauses_dict:
                        # clause does not exist in the dict
                        # -> create a blank clause to be filled later
                        clauses_dict[clause_number] = Clause()

                    clauses_dict.get(clause_number).append(token)
                    
                # 4.b. embed the clause in its passage
                
                for clause_key in clauses_dict.keys():
                    # flag for setting up the mark up
                    clause_open = False
                    clause_text_embed_in_passage = ''
                    for token in passage_token_list_item:
                        
                        if token.get('misc') is not None:
                            
                            # token is part of the current clause
                            if clause_key == token.get('misc'):
                                if not clause_open:
                                    # start the clause mark up
                                    clause_text_embed_in_passage += '<b>'
                                    clause_open = True
                            # token is NOT part of the current clause
                            else:
                                if clause_open:
                                    # close the clause mark up
                                    clause_text_embed_in_passage += '</b>'
                                    clause_open = False
                            clause_text_embed_in_passage = clause_text_embed_in_passage + ' ' + token['form']
                        # punctation
                        else:
                            clause_text_embed_in_passage = clause_text_embed_in_passage + token['form']
                            
                        # closing the mark up, in case the passage conists of a single clause
                    if clause_open:
                        clause_text_embed_in_passage += '</b>'
                    # add clause_text_embed_in_passage to the clause
                    clauses_dict.get(clause_key).clause_text_embed_in_passage = clause_text_embed_in_passage

                # 3. finally put everything together into a passage
                # Passage(sent_id, char_pos, text, clauses_dict, previous_passage, subsequent_passage)
                passage = Passage(sent_id, char_pos, passage_text,
                                  clauses_dict, previous_passage, subsequent_passage)

                passages.append(passage)
                
            # now embed the clause_text_embed_in_passage in the previous and subsequent passage
            for passage_index in range(len(passages)):
                # first passage -> only subsequent passage
                if 1 == passage_index:
                    for clause_key in passages[passage_index].clauses.keys():
                        current_clause = passages[passage_index].clauses.get(clause_key)
                        subsequent_passage = passages[passage_index+1]
                        current_clause.contextualized_text = current_clause.clause_text_embed_in_passage + subsequent_passage.text
                # last passage -> only previous passage
                elif len(passages)-1 == passage_index:
                    for clause_key in passages[passage_index].clauses.keys():
                        current_clause = passages[passage_index].clauses.get(clause_key)
                        previous_passage = passages[passage_index-1]
                        current_clause.contextualized_text = previous_passage.text + current_clause.clause_text_embed_in_passage
                # all other passages -> previous and subsequent passage
                else:
                    for clause_key in passages[passage_index].clauses.keys():
                        current_clause = passages[passage_index].clauses.get(clause_key)
                        previous_passage = passages[passage_index-1]
                        subsequent_passage = passages[passage_index+1]
                        current_clause.contextualized_text = previous_passage.text + current_clause.clause_text_embed_in_passage + subsequent_passage.text
                            
                
            documents.append(Document(document_id, passages))
            i += 1
        return documents

    def _create_passage_text(self, tokenlist):
        """
        convert TokenList to human readable text
        return: str
        """
        text = ''
        for i in range(0, len(tokenlist)):
            if 0 == i:
                text = tokenlist[i]['form']
            else:
                text += ' ' + tokenlist[i]['form']
        return text

    def _get_all_conllu_files(self, rootdir):
        """
        Scans rootdirectory and its subdirectories for all conllu files
        return: list of filenames
        """
        conllu_files = []
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                filepath = os.path.join(subdir, file)
                # _spaces.conllu leads to erros with the SPACe token. Always use _tabs.conllu
                if filepath.endswith("_tabs.conllu"):
                    conllu_files.append(filepath)
        if 0 == len(conllu_files):
            print('no conllu files found in', rootdir)
        return conllu_files

    def _field_parser(self, line, i):
        """
        return: parsed field
        """
        if '_' == line[i]:
            return None
        return line[i].split(",")
    
    def _misc_parser(self, line, i):
        """
        parsing method for the 'misc' field
        return: parsed field
        """
        if '_' == line[i]:
            return None
        try:
            int(line[i])
        except:
            raise TypeError('Could not parse misc field. Check if the other fields are parsed in the proper way and chronology')
        return int(line[i])
