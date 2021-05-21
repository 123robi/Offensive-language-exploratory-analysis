import argparse
import pandas as panda

from models.custom_classifier import main
from scripts.run_XLM import run_xlm
from scripts.run_bert import bert
from scripts.run_elmo import run_elmo


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
            dataset = "data/transformed_datasets/nova24_binary.csv"
            model = "./models/BERT/CroSloEng_FT_B_Slo"
            bert(dataset, model)
        if args.language == LANGUAGE.ENGLISH:
            # TODO ???
            pass
    if args.type == TYPE.MULTILABEL:
        if args.language == LANGUAGE.SLOVENE:
            dataset = "data/transformed_datasets/nova24_multi.csv"
            model = "./models/BERT/CroSloEng_FT_Multi_Slo_translate"
            bert(dataset, model)
        if args.language == LANGUAGE.ENGLISH:
            # TODO ???
            pass


if args.model == MODEL.ELMO:
    if args.type == TYPE.BINARY:
        if args.language == LANGUAGE.SLOVENE:
            dataset = "data/transformed_datasets/test.csv"
            weights = "models/ELMo/model_elmo_weights-B.h5"
            run_elmo(dataset, weights)
        if args.language == LANGUAGE.ENGLISH:
            # TODO ???
            pass
    if args.type == TYPE.MULTILABEL:
        if args.language == LANGUAGE.SLOVENE:
            dataset = "data/transformed_datasets/test.csv"
            weights = "models/ELMo/model_elmo_weights_multi.h5"
            run_elmo(dataset, weights)
        if args.language == LANGUAGE.ENGLISH:
            # TODO ???
            pass


if args.model == MODEL.XLM:
    if args.type == TYPE.BINARY:
        if args.language == LANGUAGE.SLOVENE:
            dataset = "data/transformed_datasets/nova24_binary.csv"
            model = "./models/XLM/XLMRoBERTa-B-Slo"
            run_xlm(dataset, model)
        if args.language == LANGUAGE.ENGLISH:
            # TODO ???
            pass
    if args.type == TYPE.MULTILABEL:
        if args.language == LANGUAGE.SLOVENE:
            dataset = "data/transformed_datasets/nova24_multi.csv"
            weights = "./models/XLM/XLMRoBERTa-Multi"
            run_xlm(dataset, weights)
        if args.language == LANGUAGE.ENGLISH:
            # TODO ???
            pass


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
    # TODO Ziga
    pass
