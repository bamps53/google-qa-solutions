from .utils import *

DATA_DATA_PATH = '../input/google-quest-challenge/'
MAX_SEQUENCE_LENGTH = 512
LEN_TEXT_COLS = 3
LEN_CATEGORY_COLS = 68
LEN_FEATURE_COLS = 9


TEXT_COLS = [
    'treated_question_title',
    'treated_question_body',
    'treated_answer'
]

CATEGORY_COLS = ['host_academia.stackexchange.com',
 'host_android.stackexchange.com',
 'host_anime.stackexchange.com',
 'host_apple.stackexchange.com',
 'host_askubuntu.com',
 'host_bicycles.stackexchange.com',
 'host_biology.stackexchange.com',
 'host_blender.stackexchange.com',
 'host_boardgames.stackexchange.com',
 'host_chemistry.stackexchange.com',
 'host_christianity.stackexchange.com',
 'host_codereview.stackexchange.com',
 'host_cooking.stackexchange.com',
 'host_crypto.stackexchange.com',
 'host_cs.stackexchange.com',
 'host_dba.stackexchange.com',
 'host_diy.stackexchange.com',
 'host_drupal.stackexchange.com',
 'host_dsp.stackexchange.com',
 'host_electronics.stackexchange.com',
 'host_ell.stackexchange.com',
 'host_english.stackexchange.com',
 'host_expressionengine.stackexchange.com',
 'host_gamedev.stackexchange.com',
 'host_gaming.stackexchange.com',
 'host_gis.stackexchange.com',
 'host_graphicdesign.stackexchange.com',
 'host_judaism.stackexchange.com',
 'host_magento.stackexchange.com',
 'host_math.stackexchange.com',
 'host_mathematica.stackexchange.com',
 'host_mathoverflow.net',
 'host_mechanics.stackexchange.com',
 'host_meta.askubuntu.com',
 'host_meta.christianity.stackexchange.com',
 'host_meta.codereview.stackexchange.com',
 'host_meta.math.stackexchange.com',
 'host_meta.stackexchange.com',
 'host_money.stackexchange.com',
 'host_movies.stackexchange.com',
 'host_music.stackexchange.com',
 'host_photo.stackexchange.com',
 'host_physics.stackexchange.com',
 'host_programmers.stackexchange.com',
 'host_raspberrypi.stackexchange.com',
 'host_robotics.stackexchange.com',
 'host_rpg.stackexchange.com',
 'host_salesforce.stackexchange.com',
 'host_scifi.stackexchange.com',
 'host_security.stackexchange.com',
 'host_serverfault.com',
 'host_sharepoint.stackexchange.com',
 'host_softwarerecs.stackexchange.com',
 'host_stackoverflow.com',
 'host_stats.stackexchange.com',
 'host_superuser.com',
 'host_tex.stackexchange.com',
 'host_travel.stackexchange.com',
 'host_unix.stackexchange.com',
 'host_ux.stackexchange.com',
 'host_webapps.stackexchange.com',
 'host_webmasters.stackexchange.com',
 'host_wordpress.stackexchange.com',
 'cat_CULTURE',
 'cat_LIFE_ARTS',
 'cat_SCIENCE',
 'cat_STACKOVERFLOW',
 'cat_TECHNOLOGY']

NEW_FEATURE_COLS = [
    'question_body_num_words',
    'answer_num_words',
    'question_vs_answer_length',
    'q_a_author_same',
    'answer_user_cat',
    'indirect',
    'question_count',
    'reason_explanation_words',
    'choice_words'
]

OUTPUT_COLS = [
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

INPUT_COLS = [
    'question_title',
    'question_body',
    'answer'
]