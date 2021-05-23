import argparse
import pandas as panda

from models.custom_classifier import main
from scripts import run_tf_idf, run_tf_idf_slovene, run_tf_idf_combined
from scripts.run_XLM import run_xlm
from scripts.run_bert import bert
from scripts.run_elmo import run_elmo
from scripts.run_elmo_multi import run_elmo_multi


class MODEL:
    BERT = "bert"
    ELMO = "elmo"
    TFIDF = "tfidf"
    CUSTOM_CLASSIFIER = "custom-classifier"
    XLM = "xlm"

class LANGUAGE:
    SLOVENE = "slovene"
    ENGLISH = "english"

class TYPE:
    BINARY = "binary"
    MULTILABEL = "multilabel"


parser = argparse.ArgumentParser(description='SiamFC Runner Script')

parser.add_argument("--model", help="Model type", required=True, action='store')
parser.add_argument("--language", help="slovene or English", required=True, action='store')
parser.add_argument("--type", help="multilabel or binary", required=True, action='store')

args = parser.parse_args()

if args.model == MODEL.BERT:
    if args.type == TYPE.BINARY:
        if args.language == LANGUAGE.SLOVENE:
            dataset = "data/datasets/nova24_binary.csv"
            model = "./models/BERT/CroSloEng_FT_B_Slo"
            bert(dataset, model, 'hatespeech')
        if args.language == LANGUAGE.ENGLISH:
            dataset = "data/datasets/english_dataset/test.csv"
            model = "./models/BERT/CroSloEng_FT_B"
            bert(dataset, model, 'hatespeech')
    if args.type == TYPE.MULTILABEL:
        if args.language == LANGUAGE.SLOVENE:
            dataset = "data/datasets/nova24_multi.csv"
            model = "./models/BERT/CroSloEng_FT_Multi_Slo"
            bert(dataset, model, 'subtype')
        if args.language == LANGUAGE.ENGLISH:
            dataset = "data/datasets/english_dataset/test.csv"
            model = "./models/BERT/CroSloEng_FT_Multi_Eng"
            bert(dataset, model, 'subtype')


if args.model == MODEL.ELMO:
    if args.type == TYPE.BINARY:
        if args.language == LANGUAGE.SLOVENE:
            # no slovene model
            pass
        if args.language == LANGUAGE.ENGLISH:
            dataset = "data/datasets/english_dataset/test.csv"
            weights = "models/ELMo/model_elmo_weights-B.h5"
            run_elmo(dataset, weights, 'hatespeech')
    if args.type == TYPE.MULTILABEL:
        if args.language == LANGUAGE.SLOVENE:
            # no slovene model
            pass
        if args.language == LANGUAGE.ENGLISH:
            dataset = "data/datasets/english_dataset/test.csv"
            weights = "models/ELMo/model_elmo_weights_multi.h5"
            run_elmo_multi(dataset, weights, 'subtype')


if args.model == MODEL.XLM:
    if args.type == TYPE.BINARY:
        if args.language == LANGUAGE.SLOVENE:
            dataset = "data/datasets/nova24_binary.csv"
            model = "./models/XLM/XLMRoBERTa-B-Slo"
            run_xlm(dataset, model, 'hatespeech')
        if args.language == LANGUAGE.ENGLISH:
            dataset = "data/datasets/english_dataset/test.csv"
            weights = "./models/XLM/XLMRoBERTa-B"
            run_xlm(dataset, weights, 'hatespeech')
    if args.type == TYPE.MULTILABEL:
        if args.language == LANGUAGE.SLOVENE:
            dataset = "data/datasets/nova24_multi.csv"
            weights = "./models/XLM/XLMRoBERTa-Multi-Slo"
            run_xlm(dataset, weights, 'subtype')
        if args.language == LANGUAGE.ENGLISH:
            dataset = "data/datasets/english_dataset/test.csv"
            weights = "./models/XLM/XLMRoBERTa-Multi"
            run_xlm(dataset, weights, 'subtype')


if args.model == MODEL.CUSTOM_CLASSIFIER:
    if args.type == TYPE.BINARY:
        if args.language == LANGUAGE.SLOVENE:
            train_dataset = panda.read_csv("data/datasets/slovene_dataset/train.csv")
            test_dataset = panda.read_csv("data/datasets/slovene_dataset/test.csv")
            dataset = test_dataset.append(train_dataset, ignore_index=True)
            main(dataset, "hatespeech")
        if args.language == LANGUAGE.ENGLISH:
            train_dataset = panda.read_csv("data/datasets/english_dataset/train.csv")
            test_dataset = panda.read_csv("data/datasets/english_dataset/test.csv")
            dataset = test_dataset.append(train_dataset, ignore_index=True)
            main(dataset, "hatespeech")

    if args.type == TYPE.MULTILABEL:
        if args.language == LANGUAGE.SLOVENE:
            train_dataset = panda.read_csv("data/datasets/slovene_dataset/train.csv")
            test_dataset = panda.read_csv("data/datasets/slovene_dataset/test.csv")
            dataset = test_dataset.append(train_dataset, ignore_index=True)
            main(dataset, "subtype")
        if args.language == LANGUAGE.ENGLISH:
            train_dataset = panda.read_csv("data/datasets/english_dataset/train.csv")
            test_dataset = panda.read_csv("data/datasets/english_dataset/test.csv")
            dataset = test_dataset.append(train_dataset, ignore_index=True)
            main(dataset, "subtype")

if args.model == MODEL.TFIDF:
    if args.type == TYPE.BINARY:

        if args.language == LANGUAGE.SLOVENE:
            train_data_si_bin = panda.read_csv("data/datasets/slovene_dataset/train.csv")
            test_data_si_bin = panda.read_csv("data/datasets/slovene_dataset/test.csv")
            frames = [train_data_si_bin, test_data_si_bin]
            combined_data_si_bin = panda.concat(frames)
            is_binary = True
            run_tf_idf_slovene.main(is_binary, combined_data_si_bin)

        if args.language == LANGUAGE.ENGLISH:
            dataset_english_bin_combined = panda.read_csv("data/datasets/english_dataset/train.csv")
            run_tf_idf_combined.main(dataset_english_bin_combined)

    if args.type == TYPE.MULTILABEL:

        if args.language == LANGUAGE.SLOVENE:
            train_data_si_mult = panda.read_csv("data/datasets/slovene_dataset/train.csv")
            test_data_si_mult = panda.read_csv("data/datasets/slovene_dataset/test.csv")
            frames = [train_data_si_mult, test_data_si_mult]
            combined_data_si_mult = panda.concat(frames)
            is_binary = False
            run_tf_idf_slovene.main(is_binary, combined_data_si_mult)

        if args.language == LANGUAGE.ENGLISH:
            dataset_t_davidson = panda.read_csv("data/datasets/english_dataset/train.csv")
            run_tf_idf.main(dataset_t_davidson)
