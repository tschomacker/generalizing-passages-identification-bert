from ml.model_util import load_model, create_data_dict
from ml.trainer import Trainer
import argparse
from transformers import AutoTokenizer
from torch import cuda

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Test a saved and fine-tuned model.')
    parser.add_argument("--test", type=str, default=None,  help="Test the model at this path.")
    parser.add_argument("--data", type=str, default=None,  help="Path to the data for test/train.")
    parser.add_argument('--labels', type=str, nargs='+', default=None, help="Labels that can be applied.")
    parser.add_argument('--keyword', type=str, default=None, help="Keyword to find the value-column in the data.")
    args = parser.parse_args()
    
    if args.test is not None:
        if args.data is None or args.labels is None:
            raise ArgumentError("--test needs --data and --labels")
        else:
            PRETRAINED_MODEL_STR = "deepset/gbert-large"
            EVALUATION_TOKENIZER = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_STR)
            
            device = 'cuda' if cuda.is_available() else 'cpu'
            labels = [label.lower() for label in args.labels]
            if args.keyword is None:
                keyword = labels[0]
            else:
                keyword = args.keyword
            
            example_model = load_model(model_path=args.test, 
                                       device=device, 
                                       petrained_model_str=PRETRAINED_MODEL_STR, 
                                       no_labels=len(labels))
            
            if 'none' in labels:
                exclude_none = True
                labels.remove('none')
            else:
                exlude_none = False

            trainer = Trainer()
            example_data_dict = create_data_dict(PRETRAINED_MODEL_STR, args.data, keyword, EVALUATION_TOKENIZER, 206, 'monaco')
            
            example_model_test_results = trainer.test(example_model,device,0.5,
                                                     example_data_dict['test'],
                                                     exclude_none, 
                                                     labels, 
                                                     False)
            print(example_model_test_results)