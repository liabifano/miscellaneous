import sys
import cPickle
from time import time

import pandas as pd
import sklearn.cross_validation as sk_cross
from sklearn.metrics import roc_auc_score

from toolz import valmap

from santander import const
from santander import trainer


def main(input_file_path, output_file_path):
    features_sets = const.FEATURES_SETS
    models = const.MODELS
    search_methods = const.SEARCH_TREE_SPACE.keys()

    all_data = pd.read_csv(input_file_path)
    train, test = sk_cross.train_test_split(all_data, test_size=const.PROP_TEST_SET, random_state=1234)

    spec_models = dict([("{}_{}_{}".format( m_name, fs_name, s), (fs, m, s))
                        for (fs_name, fs) in features_sets.iteritems()
                        for (m_name, m) in models.iteritems()
                        for s in search_methods])

    results = valmap(lambda (fs, m, s): exec_train(train, test, fs, m, s), spec_models)
    results['train'] = train
    results['test'] = test

    try:
        open(output_file_path, 'w').close()
    except IOError:
        print "Could not create file"
    with open(output_file_path, 'w') as fd:
        cPickle.dump(results, fd, protocol=2)


def exec_train(train, test, features, model, search_method):
    start = time()
    train = trainer.Trainer(model=model, data_to_train=train,
                            features=features, target=const.TARGET)

    best_params = train.find_best_params(**const.SEARCH_TREE_SPACE[search_method])
    best_model = train.estimate_model(n_estimators=100, **best_params)

    estimate_in_test = train.predict(best_model, test)
    performance_in_test = roc_auc_score(test[const.TARGET].values, estimate_in_test)

    print "%d took %s" %(time()-start, str(model))

    return {'best_model': best_model,
            'estimate_in_test': estimate_in_test,
            'performance': performance_in_test}


if __name__ == '__main__':
    main(*sys.argv[1:3])
