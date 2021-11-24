import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
from generate_adv_examples import *
import time 

NUM_PREPROCESSING_WORKERS = 2
def add_adversarial_text(example, adversarial_text):
    example['context'] = example['context'] + adversarial_text
    return example

# Incorrectly Implemented
def calc_attack_score(eval_args, dataset, string):
    eval_kwargs, model, training_args, train_dataset_featurized, tokenizer, compute_metrics_and_store_predictions, prepare_eval_dataset, trainer_class= eval_args[0], eval_args[1], eval_args[2], eval_args[3], eval_args[4], eval_args[5], eval_args[6], eval_args[7]
    # run model against subset of eval examples
    # return score as a function of (or copy of) exact match or avg f1 score

    modified_dataset = dataset.map(add_adversarial_text, fn_kwargs={"adversarial_text": string})
    modified_eval_dataset_featurized = modified_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=modified_dataset.column_names
        ) 
    
    # metric = datasets.load_metric('squad')
    trainer = trainer_class(
    model= model,
    args=training_args,
    train_dataset=train_dataset_featurized,
    eval_dataset=modified_eval_dataset_featurized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_and_store_predictions
    ) # TODO check parameters like if args is training_args
        #TODO check if train_dataset_featurized needs to be sampled as well

    results = trainer.evaluate(**eval_kwargs)
    print('Evaluation results:')
    print(results)

    raise NotImplementedError()

def generate_adv_examples(eval_args, dataset, desired_string_size, adv_vocab, beam_size):
    # scenarios = [[x] for x in adv_vocab]
    # scenarios starts for first itr starts out as size of adv_vocab, but every other iteration is size of beam
    scenarios = ["" for x in range(adv_vocab)]
    # add string to each item in scenarios
    for i in range(desired_string_size):
        for potential_word in adv_vocab:
            new_scenarios = [] # contains (adv_text, attack_score) pairs
            for scenario in scenarios:
                if scenario == "":
                    adv_text = potential_word
                else:
                    adv_text = scenario + " "+ potential_word
                
                attack_score = calc_attack_score(eval_args, dataset, adv_text)
                new_scenarios.append((adv_text, attack_score))
            scenarios = new_scenarios.sort(key = lambda x: x[1], reverse = True)[0:beam_size] # check if need reverse=True
    return scenarios

def add_adv_text_per_question(example, dictionary_mapping_example_to_index):
    question = example["question"]
    value = dictionary_mapping_example_to_index[question]
    example["context"] = example["context"] + str(value) 
    return example

def main():

    # while True:
    #     i = i+1
    argp = HfArgumentParser(TrainingArguments)
    # argp.add_argument('-f')
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], default='qa',
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default='squad',
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    argp.add_argument('--gen_adv_examples', type=bool, default=True,
                      help='Limit the number of examples to evaluate on.')
    training_args, args = argp.parse_args_into_dataclasses()
    # print('training_args: : ', training_args)
    # print('args: : ', args)
    # Dataset selection
    default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
    dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
        default_datasets[args.task]
    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}
    # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
    eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
    # Load the raw data
    dataset = datasets.load_dataset(*dataset_id)
    dataset.save_to_disk("saved_dataset")
    
    
    # Here we select the right model fine-tuning head
    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
        # prepare_eval_dataset = prepare_dataset_nli
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        # dataset_to_dictionary =  {}
        # for itr, example in enumerate(eval_dataset):
        #     dataset_to_dictionary[example['question']] = itr
        # key_0=eval_dataset[0]['question']
        # key_1 = eval_dataset[1]['question']
        # dataset_to_dictionary[key_0] = "Nazzis kill peoople"
        # dataset_to_dictionary[key_1] = "Chinese communists"
        # example 0 corresponds to Nazzis kill peoople and
        # example 1 corresponds: Chine communists
        # eval_dataset_with_adversarial_examples = eval_dataset.map(add_adv_text_per_question, fn_kwargs={"dictionary_mapping_example_to_index": dataset_to_dictionary})
        # only_diff_contexts = []
        # seen_contexts = set()
        # for example in  eval_dataset:
        #     if example['context'] in seen_contexts:
        #         continue
        #     only_diff_contexts.append(example)
        #     seen_contexts.add(example['context'])

        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )  
    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = lambda eval_preds: metric.compute(
                predictions=eval_preds.predictions, references=eval_preds.label_ids)
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = datasets.load_metric('squad')
        if args.gen_adv_examples:
            # makes the eval dataset only of length 100 when gen_adv_examples
            eval_dataset = eval_dataset.select(range(100))
            # for adv_example in adv_examples:
            beam_size = 10
            adversarial_words = set(get_common_words() + get_adv_words(100)) 
        else:
            compute_metrics = lambda eval_preds: metric.compute(
                predictions=eval_preds.predictions, references=eval_preds.label_ids)
        print("debug")
    elif args.task == 'nli':
        compute_metrics = compute_accuracy
    

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
    )
    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        if args.gen_adv_examples:
            trainer_class = QuestionAnsweringTrainer
            # arguments passed in that are necessary to test datasets with modified 'context' (where a certain adv example is appended to all 'context' in the modified dataset)
            eval_args = [eval_kwargs, model, training_args, train_dataset_featurized, tokenizer, compute_metrics_and_store_predictions, prepare_eval_dataset,trainer_class]
            adv_examples = generate_adv_examples(eval_args, eval_dataset, 5, adversarial_words, beam_size)
            # TODO: Store generated adv_examples into file
        else:
            results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), encoding='utf-8', mode='w') as f:
            json.dump(results, f)

        with open(os.path.join(training_args.output_dir, 'eval_predictions.jsonl'), encoding='utf-8', mode='w') as f:
            if args.task == 'qa':
                predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                for example in eval_dataset:
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')
            else:
                for i, example in enumerate(eval_dataset):
                    example_with_prediction = dict(example)
                    example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                    example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                    f.write(json.dumps(example_with_prediction))
                    f.write('\n')


if __name__ == "__main__":
    main()

def add_context(example, context_to_add):
    example['sentence1'] = 'My sentence: ' + example['sentence1']
    return example

updated_dataset = dataset.map(add_context)
updated_dataset['sentence1'][:5]