import yaml
from easydict import EasyDict as edict


def _get_default_config():
    c = edict()

    # dataset
    c.data = edict()
    c.data.num_folds = 5
    c.data.pseudo_path = False
    c.data.train_df_path = '../input/google-quest-challenge/train.csv'
    c.data.test_df_path =  '../input/google-quest-challenge/test.csv'
    c.data.extra_features = False
    c.data.rank_transform = False
    c.data.clean_space = False
    c.data.cutout = 0
    c.data.input_columns = [
    'question_title',
    'question_body',
    'answer'
    ]
    c.data.output_columns = [
    'question_asker_intent_understanding',
    'question_body_critical',
    'question_conversational',
    'question_expect_short_answer',
    'question_fact_seeking',
    'question_has_commonly_accepted_answer',
    'question_interestingness_others',
    'question_interestingness_self',
    'question_multi_intent',
    'question_not_really_a_question',
    'question_opinion_seeking',
    'question_type_choice',
    'question_type_compare',
    'question_type_consequence',
    'question_type_definition',
    'question_type_entity',
    'question_type_instructions',
    'question_type_procedure',
    'question_type_reason_explanation',
    'question_type_spelling',
    'question_well_written',
    'answer_helpful',
    'answer_level_of_information',
    'answer_plausible',
    'answer_relevance',
    'answer_satisfaction',
    'answer_type_instructions',
    'answer_type_procedure',
    'answer_type_reason_explanation',
    'answer_well_written'
    ]

    # model
    c.model = edict()
    c.model.num_bert = 1
    c.model.name = 'bert-base-uncased'
    c.model.max_length = 512
    c.model.batch_size = 4
    c.model.num_labels = 30
    c.model.num_heads = 1
    c.model.dropout_rate = 0.2
    c.model.output_hidden_states = False
    c.model.num_hidden_layers = 13
    c.model.siamese = False

    # train
    c.train = edict()
    c.train.batch_size = 8
    c.train.warpup_num_epoch = 0
    c.train.num_epochs = 5
    c.train.mixup_alpha = 0
    c.train.early_stop_patience = 5
    c.train.accumulation_steps = 1
    c.train.fp16 = False
    c.train.grad_clip = 100

    # test
    c.test = edict()
    c.test.batch_size = 8
    c.test.binarize = False
    c.test.num_bins = 18

    #loss
    c.loss = edict()
    c.loss.name = 'BCE'

    c.metric = edict()
    c.metric.name = 'spearmanr'
    c.metric.monitor = True
    c.metric.minimize = False

    # optimizer
    c.optimizer = edict()
    c.optimizer.name = 'AdamW'
    c.optimizer.params = edict()
    c.optimizer.params.encoder_lr = 2.0e-5
    c.optimizer.params.decoder_lr = 1.0e-3
    c.optimizer.params.weight_decay = 0

    # scheduler
    c.scheduler = edict()
    c.scheduler.name = 'plateau'
    c.scheduler.params = edict()
    c.scheduler.params.patience = 5

    c.device = 'cuda'
    c.num_workers = 0
    c.work_dir = './work_dir'
    c.project_name = 'google-qa-solutions'
    c.seed = 0

    return c


def _merge_config(src, dst):
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_path):
    with open(config_path, 'r') as fid:
        yaml_config = edict(yaml.load(fid, Loader=yaml.SafeLoader))

    config = _get_default_config()
    _merge_config(yaml_config, config)

    return config


def save_config(config, file_name):
    with open(file_name, "w") as wf:
        yaml.dump(config, wf)
