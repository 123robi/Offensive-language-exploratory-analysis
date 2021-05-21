import sys

import run_tf_idf
import run_tf_idf_combined
import run_tf_idf_slovene

if __name__ == "__main__":

    labels = 'binary'
    dataset = 'english'

    if len(sys.argv) == 2:
        labels = sys.argv[1]
    if len(sys.argv) == 3:
        dataset = sys.argv[2]

    if dataset == 'slovene':
        if labels == 'binary':
            run_tf_idf_slovene.main(True)
        elif labels == 'multilabel':
            run_tf_idf_slovene.main(False)
    elif dataset == 'english':
        if labels == 'binary':
            run_tf_idf_combined.main()
        elif labels == 'multilabel':
            run_tf_idf.main()