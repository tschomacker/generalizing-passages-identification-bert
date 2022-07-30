import time
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from xai.anchors_wrapper import AnchorsWrapper

def generate_xai_graphics(lime_clause, pipelines, labels_list, explainers, number_of_samples, coverage_samples=10000,output_dir=os.path.join('..','output','xai','GI'),verbose = False):
    """
    params:
        lime_clause : LimeClause 
        pipelines, 
        label_lists, 
        explainers
    """
    
    mpl.rcParams['axes.spines.left'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.bottom'] = False
    
    output_formats = ['pdf', 'png']
    if verbose:
        print('Start with',lime_clause.document_id, lime_clause.sent_id, lime_clause.clause_id)
    
    category = 'GI'
    
    for pipeline_t_l, labels, explainer in zip(pipelines,labels_list,explainers):
        
        anchors_wrapper = AnchorsWrapper(pipeline_t_l)
        label_vector = lime_clause.gi
        analyzed_labels = [*range(len(labels))]
        lime_clause_explainer = explainer.explain_instance(
                                                lime_clause.clause, 
                                                classifier_fn=pipeline_t_l, 
                                                num_samples=number_of_samples, 
                                                #num_features=number_of_features, 
                                                labels=analyzed_labels)

        lime_context_clause_explainer = explainer.explain_instance(
                                                lime_clause.contextualized_clause, 
                                                classifier_fn=pipeline_t_l, 
                                                num_samples=number_of_samples, 
                                                #num_features=number_of_features,
                                                labels=analyzed_labels)

        labels_assigned = []

        for label_str, label_int in zip(labels,label_vector):
            if '1' == label_int:
                labels_assigned.append(label_str)
        labels_assigned_str = ','.join(labels_assigned)


        fig_sup_title_info = lime_clause.document_id+' (Sent: '+lime_clause.sent_id+', Clause:'+lime_clause.clause_id+')'
        fig_sup_title_info = fig_sup_title_info+' (Gold '+category+'-labels: '+labels_assigned_str+')'
        
        # used for anchors
        label_index_with_highest_prob = pipeline_t_l.predict_for_anchor([lime_clause.contextualized_clause])[0]

        for label_index in tqdm(range(len(labels)), desc=category+"-Labels", disable= not verbose):
            apply_anchors = (label_index_with_highest_prob == label_index)
            
            # print normal clause
            clause_fig = lime_clause_explainer.as_pyplot_figure(label=label_index)
            clause_fig_sup_title = improve_clause_readability(lime_clause.clause) +'\n\n'+fig_sup_title_info
            plt.title(clause_fig_sup_title)
            plt.xlabel('LIME scores for: '+category+':'+labels[label_index])

            for output_format in ['pdf']:
                figure_path = os.path.join(output_dir,lime_clause.file_name+'_'+labels[label_index]+'.'+output_format) 
                clause_fig.savefig(figure_path, bbox_inches = 'tight', format= output_format)
                plt.close(clause_fig)
            # print contextualized clause
            contextualized_clause_fig = lime_context_clause_explainer.as_pyplot_figure(label=label_index)
            context_clause_fig_sup_title = improve_clause_readability(lime_clause.contextualized_clause) +'\n\n'+fig_sup_title_info
            plt.title(context_clause_fig_sup_title)

            lime_context_clause_list = lime_context_clause_explainer.as_list(label=label_index)
            lime_context_clause_list.sort(key=lambda y: abs(y[1]))
            #print('---\n',lime_context_clause_list)

            lime_context_clause_list.reverse()
            #print('---\n',lime_context_clause_list)
            #lime_context_clause_list = lime_context_clause_list[::-1]
            # print top 10
            #number_of_top_words = 10
            #lime_context_clause_top_words, lime_context_clause_top_values = zip(*lime_context_clause_list)
            
            plt.xlabel('LIME scores for: '+category+':'+labels[label_index]+'\n\n')
            
            
            if verbose:
                print('LIME for',lime_clause.document_id, lime_clause.sent_id, lime_clause.clause_id,'finished')
            # Anchor Stuff
            if apply_anchors:
                if verbose:
                    print('Start Generate Anchors for:',labels[label_index],'...')
                anchors_explanation = anchors_wrapper.generate_anchors(lime_clause.contextualized_clause,coverage_samples,verbose)
                anchors_output = 'anchors: {%s}'%';'.join([x.split(',')[0] for x in anchors_explanation.anchor])
                anchors_output = anchors_output+' with precision: %.2f' % anchors_explanation.precision
                plt.legend(loc='lower center',bbox_to_anchor = (0.5,-0.3), title=anchors_output)
                
                if verbose:
                    print('Anchors for:',labels[label_index],'finished')
                
            
            
            for number_of_top_words in [5,10,20]:
                lime_context_clause_top_words = [x[0] for x in lime_context_clause_list]
                lime_context_clause_top_values = [x[1] for x in lime_context_clause_list]
                lime_context_clause_top_words = lime_context_clause_top_words[:number_of_top_words]
                lime_context_clause_top_words.reverse()

                #print('---\n',lime_context_clause_top_words)
                lime_context_clause_top_values = lime_context_clause_top_values[:number_of_top_words]
                lime_context_clause_top_values.reverse()

                #print('---\n',lime_context_clause_top_values)
                #break
                contextualized_clause_top_fig, contextualized_clause_top_ax = plt.subplots()
                colors = []
                for value in lime_context_clause_top_values:
                    if value > 0:
                        colors.append('g')
                    else:
                        colors.append('r')
                contextualized_clause_top_ax.barh(lime_context_clause_top_words, lime_context_clause_top_values, align='center',  color=colors)
                plt.title(context_clause_fig_sup_title+'\n(only top '+str(number_of_top_words)+' by abs. value)')
                plt.xlabel('LIME scores for: '+category+':'+labels[label_index]+'\n\n')
                if apply_anchors:
                    plt.legend(loc='lower center',bbox_to_anchor = (0.5,-0.3), title=anchors_output)
                    
                for output_format in output_formats:
                    contextualized_clause_top_fig_path =   os.path.join(output_dir,lime_clause.file_name+'_context_top_'+str(number_of_top_words)+'_'+labels[label_index]+'.'+output_format)
                    contextualized_clause_top_fig.savefig(contextualized_clause_top_fig_path, bbox_inches = 'tight', format= output_format)
                    plt.close(contextualized_clause_top_fig)

            for output_format in output_formats:
                contextualized_clause_figure_path = os.path.join(output_dir,lime_clause.file_name+'_context_'+labels[label_index]+'.'+output_format) 

                contextualized_clause_fig.savefig(contextualized_clause_figure_path, bbox_inches = 'tight', format= output_format)
                plt.close(contextualized_clause_fig)
        
        
def improve_clause_readability(clause):
    new_clause = ''
    words = clause.split(' ')
    word_new_line_threshold = 8
    if word_new_line_threshold == len(words):
        return clause
    word_count = 0
    for word in words:
        if len(word) > 2:
            word_count += 1
            if word_new_line_threshold == word_count:
                new_clause += '\n'
                word_count = 0
        new_clause = new_clause + word + ' '
        
    return new_clause