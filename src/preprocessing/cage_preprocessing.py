from sitent_preprocessing import create_sitent_df
import pandas
import os
import sklearn.utils

def create_cage_df_from_dataframes(sitent_df, monaco_df, shuffle = True, verbose = True):
    """
    """
    sitent_df['gi'] = sitent_df['gi_tags_vector']
    sitent_df['root_corpus'] = 'sitent'

    sitent_df.drop(['new_sent', 'gi_tags', 'gs_tag','gi_tags_vector', 'gs_tag_vector', 'text'], axis = 1, inplace = True)
    
    
    monaco_df.drop(['comment', 'nfr'], axis = 1, inplace = True)
    monaco_df['root_corpus'] = 'monaco'

    cage_df = pandas.concat([sitent_df, monaco_df])
    if shuffle:
        cage_df = sklearn.utils.shuffle(cage_df, random_state=42)
    # invert none-label
    cage_df['generalization'] = cage_df['gi'].map(lambda x: 1 if int(x[:1]) == 0 else 0)
    
    cage_small_df = cage_df[(cage_df.root_corpus == 'monaco') |
                            (
                            (cage_df.root_corpus == 'sitent') & 
                                (
                                    (cage_df.category == 'fiction') | 
                                    (cage_df.category == 'essays')  | 
                                    (cage_df.category == 'ficlets') | 
                                    (cage_df.category == 'letter') 
                                )
                                
                            )]
    return cage_df, cage_small_df



def create_cage_df_from_files(sitent_text_path, sitent_train_test_path, monaco_path, shuffle = True, verbose = True):
    """
    """
    sitent_df = create_sitent_df(sitent_text_path, sitent_train_test_path, verbose)
    monaco_df = pandas.read_csv(monaco_path, dtype=str, delimiter='|')
    return create_cage_df_from_dataframes(sitent_df, monaco_df, verbose)

def main():
    """
    """
    output_path = os.path.join("..","..",'data') 
    sitent_text_path = os.path.join("..","..","data","sitent_gi_labels.json")
    sitent_train_test_path = os.path.join("..","..","data","sitent","annotated_corpus","train_test_split.csv")
    monaco_path = os.path.join("..","..",'data','korpus-public.csv' )
    
    
    
    cage_df, cage_small_df = create_cage_df_from_files(sitent_text_path, sitent_train_test_path, 
                                                       monaco_path, shuffle = True, verbose = True)
    cage_csv_output_path = os.path.join(output_path,"cage.csv")
    cage_small_csv_output_path = os.path.join(output_path,"cage_small.csv")
    cage_df.to_csv(cage_csv_output_path,sep='|',index=False)
    cage_small_df.to_csv(cage_small_csv_output_path,sep='|',index=False)
    
if __name__ == "__main__":
    main()