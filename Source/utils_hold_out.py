import openml
import tpot2
from tpot2.search_spaces.pipelines import ChoicePipeline, SequentialPipeline
import sklearn.metrics
import sklearn
from sklearn.metrics import (roc_auc_score, log_loss)
import traceback
import dill as pickle
import os
import time
import numpy as np
import sklearn.model_selection
from functools import partial

# generate scores for selection schemes to use
def selection_objectives(est,X,y,X_select,y_select,classification):
    # fit model
    est.fit(X,y)

    if classification:
        return list(est.predict(X_select) == y_select)

    else:
        # regression: wanna minimize the distance between them
        return list(np.absolute(y_select - est.predict(X_select)))

# generate scores for selection schemes to use
def selection_objectives_accuracy(est,X,y,X_select,y_select,classification):
    # fit model
    est.fit(X,y)

    if classification:
        return [[float(sum(list(est.predict(X_select) == y_select))) / float(len(y_select))], [float(tpot2.objectives.complexity_scorer(est,0,0))]]

    else:
        # regression: wanna minimize the distance between them
        return list(np.absolute(y_select - est.predict(X_select)))

# calculate complexity for tpot to track in evaluated individuals
def sampling_complexity(est,X,y):
    # fit model
    est.fit(X,y)
    return [float(tpot2.objectives.complexity_scorer(est,0,0))]

def get_selection_scheme(scheme, classification):
    if scheme == 'random':
        return tpot2.selectors.random_selector
    elif scheme == 'lexicase' and classification:
        return tpot2.selectors.lexicase_selection
    elif scheme == 'lexicase' and not classification:
        # todo: add lexicase selection for regression (ask Anil for his implementation)
        return tpot2.selectors.lexicase_selection_regression
    elif scheme == 'tournament':
        return tpot2.selectors.tournament_selection
    else:
        raise ValueError(f"Unknown selection scheme: {scheme}")

# generate pipeline search space: selector -> transformer -> selector -> regressor/classifier
def get_pipeline_space(classification=True):
    # selector -> transformer -> selector -> regressor/classifier

    # if classification problem
    if classification:
        return tpot2.search_spaces.pipelines.SequentialPipeline([
            tpot2.config.get_search_space(["selectors_classification","Passthrough"]),
            tpot2.config.get_search_space(["transformers","Passthrough"]),
            tpot2.config.get_search_space(["selectors_classification","Passthrough"]),
            tpot2.config.get_search_space("classifiers")])
    else:
        return tpot2.search_spaces.pipelines.SequentialPipeline([
            tpot2.config.get_search_space(["selectors_regression","Passthrough"]),
            tpot2.config.get_search_space(["transformers","Passthrough"]),
            tpot2.config.get_search_space(["selectors_regression","Passthrough"]),
            tpot2.config.get_search_space("regressors")])

# get estimator parameters
def get_estimator_params(n_jobs,
                         classification,
                         scheme,
                         split,
                         X_train,
                         y_train,
                         seed):
    # split data
    print('(train)', 1.0-split, '/ (select)', split)
    if classification:
            X_learn, X_select, y_learn, y_select = sklearn.model_selection.train_test_split(X_train, y_train, train_size=1.0-split, test_size=split, stratify=y_train, random_state=seed)
    else:
        X_learn, X_select, y_learn, y_select = sklearn.model_selection.train_test_split(X_train, y_train, train_size=1.0-split, test_size=split, random_state=seed)
    print('X_learn:',X_learn.shape,'|','y_learn:',y_learn.shape)
    print('X_select:',X_select.shape,'|','y_select:',y_select.shape)

    # create selection objective functions
    select_objective = partial(selection_objectives_accuracy,X=X_learn,y=y_learn,X_select=X_select,y_select=y_select,classification=classification)
    select_objective.__name__ = 'sel-obj'

    # return dictionary based on selection scheme we are using
    return {
        # evaluation criteria
        'scorers': [],
        'scorers_weights':[],
        'cv': sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=seed), # not used
        'other_objective_functions': [select_objective],
        'other_objective_functions_weights': [1.0, -1.0],
        'objective_function_names': ['performance', 'complexity'],

        # evolutionary algorithm params
        'population_size' : 10,
        'generations' : 3,
        'n_jobs':n_jobs,
        'survival_selector' :None,
        'parent_selector': get_selection_scheme(scheme, classification),
        'random_state': seed,

        # offspring variation params
        'mutate_probability': 0.7,
        'crossover_probability': 0.0,
        'crossover_then_mutate_probability': 0.3,
        'mutate_then_crossover_probability': 0.0,

        # estimator params
        'memory_limit':0,
        'preprocessing':False,
        'classification' : True,
        'verbose':3,
        'max_eval_time_seconds':60*5,
        'max_time_seconds': float("inf"),

        # pipeline search space
        'search_space': get_pipeline_space(classification),
        }

# get test scores
def score(est, X, y, train_performance, complexity, classification):

    if classification:
        # get classification testing score
        performance = np.float32(sklearn.metrics.get_scorer("accuracy")(est, X, y))
    else:
        # regression testing score
        performance = 0
    return {
            'testing_performance': performance,
            'training_performance': train_performance,
            'complexity': complexity
    }

#https://github.com/automl/ASKL2.0_experiments/blob/84a9c0b3af8f7ac6e2a003d4dea5e6dce97d4315/experiment_scripts/utils.py
def load_task(task_id, preprocess=True):

    cached_data_path = f"data/{task_id}_{preprocess}.pkl"
    print(cached_data_path)
    if os.path.exists(cached_data_path):
        d = pickle.load(open(cached_data_path, "rb"))
        X_train, y_train, X_test, y_test = d['X_train'], d['y_train'], d['X_test'], d['y_test']
    else:
        task = openml.tasks.get_task(task_id)


        X, y = task.get_X_and_y(dataset_format="dataframe")
        train_indices, test_indices = task.get_train_test_split_indices()
        X_train = X.iloc[train_indices]
        y_train = y.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_test = y.iloc[test_indices]

        if preprocess:
            preprocessing_pipeline = sklearn.pipeline.make_pipeline(tpot2.builtin_modules.ColumnSimpleImputer("categorical", strategy='most_frequent'), tpot2.builtin_modules.ColumnSimpleImputer("numeric", strategy='mean'), tpot2.builtin_modules.ColumnOneHotEncoder("categorical", min_frequency=0.001, handle_unknown="ignore"))
            X_train = preprocessing_pipeline.fit_transform(X_train)
            X_test = preprocessing_pipeline.transform(X_test)


            le = sklearn.preprocessing.LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)

            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

            if task_id == 168795: #this task does not have enough instances of two classes for 10 fold CV. This function samples the data to make sure we have at least 10 instances of each class
                indices = [28535, 28535, 24187, 18736,  2781]
                y_train = np.append(y_train, y_train[indices])
                X_train = np.append(X_train, X_train[indices], axis=0)

            d = {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test}
            if not os.path.exists("data"):
                os.makedirs("data")
            with open(cached_data_path, "wb") as f:
                pickle.dump(d, f)

    return X_train, y_train, X_test, y_test

# execute task with tpot2
def execute_experiment(split, scheme, task_id, n_jobs, save_path, seed, classification):
    # generate directory to save results
    save_folder = f"{save_path}/{seed}-{task_id}"
    if not os.path.exists(save_folder):
        print('CREATING FOLDER:', save_folder)
        os.makedirs(save_folder)
    else:
        print('FOLDER ALREADY EXISTS:', save_folder)
        return

    # run experiment
    try:
        print("LOADING DATA")
        X_train, y_train, X_test, y_test = load_task(task_id, preprocess=True)

        # get estimator parameters
        est_params = get_estimator_params(n_jobs=n_jobs,classification=classification,scheme=scheme,split=split,X_train=X_train,y_train=y_train,seed=seed)

        est = tpot2.TPOTEstimator(**est_params)

        start = time.time()
        print("ESTIMATOR FITTING")
        est.fit(X_train, y_train)
        duration = time.time() - start
        print("ESTIMATOR FITTING COMPLETE:", duration, 'seconds')

        #filter evaluated_individuals pandas dataframe with individuals best performance
        best_performers = est.evaluated_individuals[est.evaluated_individuals['performance'] == est.evaluated_individuals['performance'].max()]

        # filter by the smallest complexity
        best_performers = best_performers[best_performers['complexity'] == best_performers['complexity'].min()]

        # randomly select one of the best performers with seed set for reproducibility
        best_performer = best_performers.sample(1, random_state=seed)

        # get best performer performance and cast to numpy float32
        performance = np.float32(best_performer['performance'].values[0])

        # get best performer complexity and cast to numpy uint64
        complexity = np.uint64(best_performer['complexity'].values[0])

        results = score(est, X_test, y_test, performance, complexity, classification)
        print('res')
        print('type(res):', type(results))

        print('*'*50)
        for k,v in results.items():
            print(k, '(', type(k) ,')',':',v,'(', type(v) ,')')
            print('*'*50)


        results["task_id"] = task_id
        results["selection"] = scheme
        results["seed"] = seed

        print('SAVING:SCORES.PKL')
        with open(f"{save_folder}/scores.pkl", "wb") as f:
            pickle.dump(results, f)

    except Exception as e:
        trace =  traceback.format_exc()
        pipeline_failure_dict = {"task_id": task_id, "selection": scheme, "seed": seed, "error": str(e), "trace": trace}
        print("failed on ")
        print(save_folder)
        print(e)
        print(trace)

        with open(f"{save_folder}/failed.pkl", "wb") as f:
            pickle.dump(pipeline_failure_dict, f)

    return