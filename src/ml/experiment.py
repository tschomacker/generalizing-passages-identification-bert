from scipy.stats import zscore
from tqdm.notebook import tqdm
from ml.trainer import Trainer
from ml.multi_label_classification_model import MultiLabelClassificationModel
import datetime
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import pandas as pd
import matplotx
import warnings

class Experiment:
    """
    Utility class to read, write and plot results from experiments.
    """
    def __init__(self, name, output_dir='output'):        
        self.keys = None
        self.delimiter = ';'
        file_name = name+'_'+datetime.datetime.now().strftime("%y%m%d")+".csv"
        self.file_name = os.path.join('..',output_dir,file_name)
        self.results = []
        self.results_df = pd.DataFrame()
        
    def _string_to_list_of_numerics(self,element):
        """
        converts an string to a list<numeric>, when it is possible
        """
        if isinstance(element, str):
            if '[' == element[0]:
                # contains floats
                if '.' in element:
                    element = [float(x) for x in element.replace('[','').replace(']','').split(',')]
                else:
                    element = [int(x) for x in element.replace('[','').replace(']','').split(',')]
        return element
    
        
    def append(self, experiment_or_path):
        """
        inplace append another experiments results to this experiment
        by concating the dataframes
        params
            experiment_or_path 
        """
        if Experiment == type(experiment_or_path):
            additional_results_df = experiment_or_path.results_df
        if pd.DataFrame == type(experiment_or_path):
            additional_results_df = experiment_or_path
        elif os.path.exists(experiment_or_path):
            loaded_experiment = Experiment.read_csv(experiment_or_path)
            additional_results_df = loaded_experiment.results_df
        else:
            raise Exception(experiment_or_path, "is not a valid input")
        self.results_df = pd.concat([self.results_df, additional_results_df], ignore_index=True)
    
    def read_csv(csv_path):
        """
            Create an Experiment from csv
            TODO
        """
        
        experiment = Experiment(name='')
        read_df = pd.read_csv(csv_path, dtype=str, delimiter='|')
        # heres needs to happen some preprocessing
        # parse all columns
        for key in read_df:
            read_df[key] = read_df[key].apply(lambda x: experiment._string_to_list_of_numerics(x)) 
        experiment.results_df = read_df
        experiment.file_name = csv_path
        return experiment

        
    
    def execute_runs(self,params):
        """
        Execute multiple train and validation runs via iterating over all paramter lists in params.
        params:
            params : dict
        return : 
            results: list of dicts with results 
        """
        experiments = []
        
        # intialize progess bar
        no_experiments = len(params['data'])*len(params['lr'])*len(params['thresholds'])
        no_experiments = no_experiments*len(params['loss_weights'])*len(params['optimizer'])
        no_experiments = no_experiments*len(params['loss_func'])*len(params['dropout_hidden'])*len(params['dropout_attention'])
        pbar = tqdm(total=no_experiments,desc="Experiments ("+params['name']+")")
        
        
        # iterate over all parameters
        for learning_rate in params['lr']:
            for optimizer in params['optimizer']:
                for loss_func in params['loss_func']:
                    for loss_weights in params['loss_weights']:
                        for threshold in params['thresholds']:
                            for data_dict in params['data']:
                                for dropout_hidden in params['dropout_hidden']:
                                    for attention_dropout in params['dropout_attention']:
                                        experimental_model = MultiLabelClassificationModel(pre_trained_model=data_dict['pretrained'], 
                                                                                           number_of_labels=params['no_labels'],
                                                                                           hidden_dropout_prob= dropout_hidden,
                                                                                           attention_probs_dropout_prob=attention_dropout
                                                                                          )
                                        experimental_model.to(params['device'])
                                        trainer = Trainer()
                                        if 'multi'==data_dict['task']:
                                            exclude_none = True
                                        elif 'binary'==data_dict['task']:
                                            exclude_none = False

                                        trained_model, experimental_results = trainer.train(experimental_model, 
                                                                                params['device'],
                                                                                params['epochs'], 
                                                                                optimizer, 
                                                                                learning_rate,
                                                                                threshold,
                                                                                data_dict['train'], 
                                                                                data_dict['validate'],
                                                                                loss_func,
                                                                                loss_weights,
                                                                                exclude_none,
                                                                                None,#early_stopping_config
                                                                                params['verbose'])

                                        # prepare the results
                                        experimental_results['batch_size'] = data_dict['batch_size']
                                        experimental_results['task'] = data_dict['task']
                                        experimental_results['corpus'] = data_dict['corpus']
                                        experimental_results['dropout_hidden'] = dropout_hidden
                                        experimental_results['dropout_attention'] = attention_dropout
                                        experimental_results['TEST-F1-macro'] = trainer.test(trained_model, 
                                                                                             params['device'], 
                                                                                             threshold, 
                                                                                             data_dict['test'],
                                                                                             exclude_none,
                                                                                             None, 
                                                                                             params['verbose'])['F1-macro']
                                        self.results.append(experimental_results)
                                        # switching to df instead of dict
                                        self.results_df = pd.concat([self.results_df, pd.DataFrame([experimental_results])],
                                                                    ignore_index=True)
                                        #self.results_df.to_csv(self.file_name, sep='|', index=False)
                                        self.to_csv()
                                        pbar.update(1)

        pbar.close()
        return self.results_df
    
    def to_csv(self, output_file=None):
        if output_file is None:
            output_file = self.file_name
        self.results_df.to_csv(output_file, sep='|', index=False)
        
    
    def _format_loss_weights(self, weights):
        """
        Truncates floats in a list and formats the complete list into a string.
        """
        weights_list = self._string_to_list_of_numerics(weights)
        if isinstance(weights_list, list):
            weights_list = [ str(weight)[:4] for weight in weights_list]
            weights_list = [ float(weight) for weight in weights_list]
            weights_list = str(weights_list)
        return weights_list
    
    def plot(self, title, attributes, observed_parameters, export_file_name, form=None, show=False, benchmarks={'f1_score_weighted':1}):
        """
        generates parameterized plots for the experiments
        params:
            title : string that will be the title of the plots
            experiments : dict with the results of the experiments.
            attributes : list of string attributes that will be plotted individually.
            export_file_name : name of the exported file.
            form : string that determines the plot dimensions.
            show : boolean indicates whether it should be displayed
            benchmarks: dict of attributes and a benchmark value
        """
        experiments = self.results_df
        epochs = experiments['epochs'][0]
        plt.style.use('default')
        # https://www.tutorialspoint.com/matplotlib-plot-lines-with-colors-through-colormap
        COLORS = plt.cm.Paired(np.linspace(0, 1, len(experiments)))
        
        markers = ['o', 's', '^', "d", "*", 'v', 's' ]
        markers = markers + markers +  markers + markers + markers + markers + markers + markers + markers
        legend_location = "lower left"
        ncol = 1
        bbox_to_anchor=(1,0)

        if form == 'vertical':
            dimensions = (9, 12)
        elif form == 'square':
            dimensions = (9, 9)
            legend_location = "upper right"
            #bbox_to_anchor=(0.5, -0.75)
            
            legend_fontsize = 17#'large'
            matplotlib.rc('xtick', labelsize=legend_fontsize) 
            matplotlib.rc('ytick', labelsize=legend_fontsize)
            matplotlib.rc('axes', labelsize=legend_fontsize)
        elif form == 'horizontal':
            dimensions = (12, 9)
        elif form == 'horizontal-flat':
            dimensions = (12, 4)
            ncol = 3
            bbox_to_anchor = (1,-0.1)
        elif form == 'small':
            ncol = 1
            dimensions = (6, 2)
            legend_location
        else:
            warnings.warn(str(form)+' is not a valid form-format for visulization')
            return None

        for attribute in attributes:
            fig = plt.figure(figsize=dimensions)
            if attribute in benchmarks:
                #goal = [benchmarks[attribute] for _ in self.results[0]['epochs']]
                goal = [benchmarks[attribute]] * len(epochs)
                plt.plot(epochs, goal, label='benchmark', color='black', linestyle='dashed')
            #for experiment, marker, color in zip(experiments.iterrows(), markers, COLORS):
            max_y_tick = 0
            for index, experiment in experiments.iterrows():
                marker = markers[index]
                color = COLORS[index]
                y_values = experiment[attribute]
                if max_y_tick < max(y_values):
                    max_y_tick = max(y_values)
                plt.plot(epochs,y_values, label=self._generate_inline_label(experiment,observed_parameters), color=color, marker=marker)
            plt.ylabel(attribute)
            plt.xlabel('epoch')
            plt.xticks(epochs)
            y_tick_step_size = 0.025
            max_y_tick = max_y_tick + y_tick_step_size
            plt.yticks(np.arange(0, max_y_tick, y_tick_step_size))
            
            use_inline_label = False
            if use_inline_label:
                matplotx.line_labels()
                plt.figtext(x=0.5,y=0.00, s=self._generate_fig_text(experiments, observed_parameters), 
                        ha="center", va="center", bbox={"facecolor":"grey", "alpha":0.1}, fontsize=legend_fontsize)
            else:
                plt.legend(loc='lower center', bbox_to_anchor=bbox_to_anchor, ncol= ncol, 
                           title=self._generate_fig_text(experiments, observed_parameters), 
                           title_fontsize=legend_fontsize, fontsize=legend_fontsize)
            
            
            fig.savefig("../output/"+export_file_name+'_'+attribute+".pdf", bbox_inches = 'tight')
            if show:
                plt.show
            else:
                plt.close(fig)
                
    def _generate_fig_text(self,experiments,exclude_parameters):
        experiment = experiments.iloc[0]
        plot_label = ''
        if 'pretrained' not in exclude_parameters:
            plot_label = plot_label+experiment['pretrained']+' '
        if 'corpus' not in exclude_parameters:
            plot_label = plot_label+'corpus: '+experiment['corpus']
        plot_label = plot_label+' task: '+experiment['task']+'\noptim.: '
        
        if 'optimizer' not in exclude_parameters:
            plot_label = plot_label+experiment['optimizer']
        if 'lr' not in exclude_parameters:
            plot_label = plot_label+' lr: '+str(experiment['lr'])
        plot_label = plot_label+ ' threshold:'+str(experiment['threshold'])+' batch_size:'+str(experiment['batch_size'])
        plot_label = plot_label+'\nloss:'+str(experiment['loss_func'])+' weights:'+self._format_loss_weights(experiment['loss_weights'])
        if 'dropout_hidden' not in exclude_parameters:
            plot_label = plot_label+'\ndropout_hidden:'+str(experiment['dropout_hidden'])
        if 'dropout_attention' not in exclude_parameters: 
            plot_label = plot_label+' dropout_attention:'+str(experiment['dropout_attention'])
        return plot_label
        
        
    def _generate_inline_label(self,experiment,include_parameters):
        inline_label = ''
        for parameter in include_parameters:
            if parameter not in ['optimizer', 'corpus','pretrained']:
                text = parameter + ':'+ str(experiment[parameter])
            else:
                text = str(experiment[parameter])
            inline_label = inline_label + ' '+ text
        return inline_label
        