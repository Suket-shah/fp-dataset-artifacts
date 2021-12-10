import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy, compute_distance_off, get_common_words, get_charged_words
import time 
import random
import os
import json
# import logging
# import copy
# import re


# Parameters for generating adversarial text
global eval_subset_size
global adv_dict_size
global filter_questions_based_on_acceptable_question_types
global only_use_common_words_small
global acceptable_question_types
global error_cnt

eval_subset_size = 20
universal_trigger_len = 10
only_use_common_words_small = False
filter_questions_based_on_acceptable_question_types = True 
# acceptable_question_types = ["Why","Who", "When", "Where", "How"] 
adv_dict_size= 200
error_cnt = 0
beam_size = 8

if beam_size>adv_dict_size:
    beam_size = adv_dict_size


NUM_PREPROCESSING_WORKERS = 2

# file name is either all_words.txt or beam_best.txt
def write_adv_text(adv_examples, file_name):
    file1 = open(file_name, "a")  # append mode
    file1.write("\n")
    for example in adv_examples:
        file1.write(str(example) +"\n")
    file1.write("\n")
    file1.close()

def get_cross_entropy_loss(universal_trigger_string, trainer_args):
    trainer_class, model, training_args, train_dataset_featurized, eval_dataset, tokenizer, compute_metrics_and_store_predictions, prepare_eval_dataset = trainer_args
    # STEP 1: modify eval dataset to include universal_trigger_string plus target span "To kill american people" with map method
    adv_text = " " + universal_trigger_string + " To kill american people."
    modified_dataset = eval_dataset.map(add_adversarial_text, fn_kwargs={"adversarial_text": adv_text})

    # Step 2: Create trainer with modified dataset that is featurized 
    # Step 2A: featurize modified_dataset
    modified_eval_dataset_featurized = modified_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=modified_dataset.column_names
        ) 
    trainer = trainer_class(
    model= model,
    args=training_args,
    train_dataset=train_dataset_featurized,
    eval_dataset=modified_eval_dataset_featurized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_and_store_predictions
    )
    eval_kwargs = {}
    eval_kwargs['eval_examples'] = modified_dataset
    # Step 3: Call evaluate with correct kwargs
    results = trainer.evaluate(**eval_kwargs)
    total_loss = results['eval_CrossEntropyLoss'] #TODO: suket Check if the loss is being calculated correctly
    return total_loss

def generate_universal_triggers(universal_trigger_len, all_possible_words, beam_size, trainer_args):
    # scenarios = [("",0)] # for x in adv_vocab
    scenarios = []
    for i in range(universal_trigger_len):
        t1=time.time()
        if i==0:
            # universal_trigger_list = ["the" for x in range(universal_trigger_len)]
            for word in all_possible_words:
                # universal_trigger_list[0] = word
                # universal_trigger_string = ' '.join(universal_trigger_list)
                universal_trigger_string = word + " "
                total_loss = get_cross_entropy_loss(universal_trigger_string, trainer_args) # TODO pass in necessary parameters
                # TODO: ensure that list is being copied by value and not by memory reference OR might have to use copy python library
                # scenarios.append((copy.deepcopy(universal_trigger_list), total_loss))
                scenarios.append( (universal_trigger_string, total_loss) )
            scenarios.sort(key = lambda x: x[1])
            # all_words.txt or beam_best.txt
            # trim scenarios to length of beam size
            write_adv_text(scenarios, "generated_triggers/all_words.txt")
            if len(scenarios)>beam_size:
                scenarios = scenarios[0:beam_size] # check if need reverse=True
            write_adv_text(scenarios, "generated_triggers/beam_best.txt")
            t2=time.time()
            write_adv_text(["Iteration 0 took " + str(t2-t1)], "generated_triggers/all_words.txt")
            write_adv_text(["Iteration 0 took " + str(t2-t1)], "generated_triggers/beam_best.txt")
        else:
            # use beam search 
            previous_k_best_universal_triggers = [scenarios[i][0] for i in range(len(scenarios))]
            new_scenarios = []
            t1=time.time()
            for prev_string in previous_k_best_universal_triggers:
                t3 = time.time()
                for count, word in enumerate(all_possible_words):
                    universal_trigger_string = prev_string + word + " "
                    # universal_trigger_string = ' '.join(prev_word_seq)
                    total_loss = get_cross_entropy_loss(universal_trigger_string, trainer_args)
                    # TODO: ensure that list is being copied by value and not by memory reference OR might have to use copy python library
                    new_scenarios.append((universal_trigger_string, total_loss))
                    # if count%100==0 and count!=0:
                    #     t4=time.time()
                    #     write_adv_text(["100 words took " + str(t4-t3)], "all_words.txt")
                    #     t3=time.time()
                # log_progress(scenarios)
            t2=time.time()
            write_adv_text(["Iteration " +str(i)+" took " + str(t2-t1)],"generated_triggers/all_words.txt")
            write_adv_text(["Iteration " +str(i)+" took " + str(t2-t1)],"generated_triggers/beam_best.txt")
            scenarios = sorted(new_scenarios, key = lambda x: x[1])
            write_adv_text(scenarios, "generated_triggers/all_words.txt")
            # trim scenarios to length of beam size
            if len(scenarios)>beam_size:
                scenarios = scenarios[0:beam_size] # check if need reverse=True
            write_adv_text(scenarios, "generated_triggers/beam_best.txt")

def add_adversarial_text(example, adversarial_text):
    example['context'] = example['context'] + adversarial_text
    return example

def main():
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
                      default='trained_model/checkpoint-32500',
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
    argp.add_argument('--use_subset_train_examples', type=bool, default=True,
                        help='Limit the number of examples to train on.')
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
            # only select questions that are why 
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )

    if training_args.do_eval:
        eval_dataset = dataset[eval_split]

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
    # eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    # compute_metrics = lambda eval_preds: metric.compute(
    #             predictions=eval_preds.predictions, references=eval_preds.label_ids)
    compute_metrics = compute_distance_off
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer

        if args.gen_adv_examples:
            # makes the eval dataset only of length 100 when gen_adv_examples
            if filter_questions_based_on_acceptable_question_types:
                # acceptable_question_types = ["Why", "Where", "When"]
                # eval_dataset = eval_dataset.filter(lambda example: [ele for ele in acceptable_question_types if(ele in example['question'])] != [] ) 

                acceptable_question_types1 = ["Why"]
                acceptable_question_types2 = ["What"]
                acceptable_question_types3 = ["When"]
                acceptable_question_types4 = ["Where"]
                acceptable_question_types5 = ["How"]
                acceptable_question_types = acceptable_question_types1 + acceptable_question_types2 + acceptable_question_types3 + acceptable_question_types4 + acceptable_question_types5

                eval_dataset0 = eval_dataset.filter(lambda example: [ele for ele in acceptable_question_types1 if(ele in example['question'])] != [] )  
                eval_dataset0 = eval_dataset0.select(range(int(eval_subset_size/5)))
                eval_dataset1 = eval_dataset.filter(lambda example: [ele for ele in acceptable_question_types2 if(ele in example['question'])] != [] )  
                eval_dataset1 = eval_dataset1.select(range(int(eval_subset_size/5)))
                eval_dataset2 = eval_dataset.filter(lambda example: [ele for ele in acceptable_question_types3 if(ele in example['question'])] != [] )  
                eval_dataset2 = eval_dataset2.select(range(int(eval_subset_size/5)))
                eval_dataset3 = eval_dataset.filter(lambda example: [ele for ele in acceptable_question_types4 if(ele in example['question'])] != [] )  
                eval_dataset3 = eval_dataset3.select(range(int(eval_subset_size/5)))
                eval_dataset4 = eval_dataset.filter(lambda example: [ele for ele in acceptable_question_types5 if(ele in example['question'])] != [] )  
                eval_dataset4 = eval_dataset4.select(range(int(eval_subset_size/5)))
                dataset_list = [eval_dataset0, eval_dataset1, eval_dataset2, eval_dataset3, eval_dataset4]
                total_dataset = datasets.concatenate_datasets(dataset_list)
                eval_dataset = total_dataset # TODO 12-4-21: instead of entire range, pick indices better
                eval_dataset = eval_dataset.shuffle(seed=42)
            
            common_words_small = get_common_words("vocabulary_sets/common_words.txt")
            common_words_large = get_common_words("vocabulary_sets/1000_common_words.txt")
            charged_words = get_charged_words(500)
            
            more_words = ["certainly", "maybe", "probably", "everything", "due", "reason", "consequently"]
            # We want all common_words_small, and a mix of common_words_large and charged_words
            charged_words_sampled = random.sample(charged_words, int(adv_dict_size/4))
            common_words_sampled= random.sample(common_words_large, int(adv_dict_size/2))
            where_and_when = get_common_words("vocabulary_sets/where_and_when.txt")
            if only_use_common_words_small:
                adversarial_words = list(set(common_words_small))
            else:
                adversarial_words = list(set(charged_words_sampled + common_words_sampled + common_words_small + where_and_when))
                adversarial_words.extend(more_words)
            adversarial_words = list(set(adversarial_words))
        else:
            compute_metrics = lambda eval_preds: metric.compute(
                predictions=eval_preds.predictions, references=eval_preds.label_ids)
        # eval_kwargs['eval_examples'] = eval_dataset
        metric = datasets.load_metric('squad')

    elif args.task == 'nli':
        compute_metrics = compute_accuracy
    
    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Train and/or evaluate
    if training_args.do_train:
        trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset_featurized,
        eval_dataset=eval_dataset_featurized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_and_store_predictions
        )
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
            # eval_args = [eval_kwargs, model, training_args, train_dataset_featurized, tokenizer, compute_metrics_and_store_predictions, prepare_eval_dataset,trainer_class]
            # word_phrases = ["by", "for the fact that", "in view of the fact that", "the number 42"]

            # TODO: be more clever about which eval examples and questions to use: perhaps use filter method or something else
            trainer_args = [trainer_class, model, training_args, train_dataset_featurized, eval_dataset, tokenizer, compute_metrics_and_store_predictions, prepare_eval_dataset]
            parameters_string = ["eval_subset_size: " + str(eval_subset_size) + ", adv_dict_size: " + str(adv_dict_size) + ", filter_questions_based_on_acceptable_question_types:" + str(filter_questions_based_on_acceptable_question_types) + ", acceptable_question_types: " + str(acceptable_question_types)]
            write_adv_text(parameters_string, "generated_triggers/beam_best.txt")
            write_adv_text(parameters_string, "generated_triggers/all_words.txt")
            generate_universal_triggers(universal_trigger_len, adversarial_words, beam_size, trainer_args)
            
if __name__ == "__main__":
    main()

