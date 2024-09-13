import openml
import tpot2
from tpot2.search_spaces.pipelines import ChoicePipeline, SequentialPipeline
import sklearn.metrics
import sklearn
import traceback
import dill as pickle
import os
import time
import numpy as np
import sklearn.model_selection
from functools import partial

# lexicase selection with ignoring the complexity column
def lexicase_selection_no_comp(scores, k, rng=None, n_parents=1,):
    """Select the best individual according to Lexicase Selection, *k* times.
    The returned list contains the indices of the chosen *individuals*.
    :param scores: The score matrix, where rows the individulas and the columns are the corresponds to scores on different objectives.
    :returns: A list of indices of selected individuals.
    This function uses the :func:`~random.choice` function from the python base
    :mod:`random` module.
    """
    rng = np.random.default_rng(rng)
    chosen =[]

    # for debugging
    # make sure the first column to the second to last one is either True or False
    # assert np.all(np.isin(score[:-2], [np.float32(True), np.float32(False)]) for score in scores)
    # make sure the last column is greater than 1
    # assert np.all(score[-1] > 1 for score in scores)

    for i in range(k*n_parents):
        candidates = list(range(len(scores)))
        cases = list(range(len(scores[0]) - 1)) # ignore the last column which is complexity
        rng.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            best_val_for_case = max(scores[candidates,cases[0]])
            candidates = [x for x in candidates if scores[x, cases[0]] == best_val_for_case]
            cases.pop(0)
        chosen.append(rng.choice(candidates))

    return np.reshape(chosen, (k, n_parents))
# generate set of unaggreated selection objectives
def unaggregated_selection_objectives(est,X,y,cv,classification):
    # hold all the scores
    scores = []
    complexity = []

    for train_index, test_index in cv.split(X, y):
        # make a copy of the estimator
        this_fold_pipeline = sklearn.base.clone(est)

        # get data split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # fit model
        this_fold_pipeline.fit(X_train, y_train)

        # get complexity score
        complexity += [tpot2.objectives.complexity_scorer(this_fold_pipeline,0,0)]

        # grade predictions on test set
        if classification:
            scores += list(np.bool_(this_fold_pipeline.predict(X_test) == y_test))
        else:
            scores += list(np.absolute(y_test - this_fold_pipeline.predict(X_test), dtype=np.float32))

        del this_fold_pipeline
        del X_train
        del X_test
        del y_train
        del y_test

    # append complexity to scores
    scores.append(np.mean(complexity, dtype=np.float32))

    # return scores
    assert len(scores) == len(y)+1
    return scores

# generate set of aggregated selection objectives
def aggregated_selection_objectives(est,X,y,cv_splits,classification):
    # hold all the scores
    scores = []
    complexity = []

    for train_index, test_index in cv_splits:
        # get data split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # fit model
        est.fit(X_train, y_train)

        # get complexity score
        complexity += [tpot2.objectives.complexity_scorer(est,0,0)]

        # grade predictions on test set
        if classification:
            scores += [np.sum(list(np.bool_(est.predict(X_test) == y_test)), dtype=np.uint32)]
        else:
            scores += [np.sum(list(np.absolute(y_test - est.predict(X_test), dtype=np.float32)), dtype=np.float32)]

    # append complexity to scores
    scores.append(np.mean(complexity, dtype=np.float32))

    # return scores
    assert len(scores) == 11
    return scores

# generate set of compressed selection objectives
def compressed_selection_objectives(est,X,y,cv_splits,classification):
    # hold all the scores
    scores = []
    complexity = []

    for train_index, test_index in cv_splits:
        # get data split
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # fit model
        est.fit(X_train, y_train)

        # get complexity score
        complexity += [tpot2.objectives.complexity_scorer(est,0,0)]

        # grade predictions on test set
        if classification:
            scores += [np.mean(np.bool_(est.predict(X_test) == y_test))]


        else:
            scores += [np.mean(list(np.absolute(y_test - est.predict(X_test), dtype=np.float32)), dtype=np.float32)]

    # make sure we have the right number of scores
    assert len(scores) == 10
    assert len(complexity) == 10

    # return cross validation scores and complexity score
    return np.mean(scores, dtype=np.float32), np.mean(complexity, dtype=np.float32)

# get selection scheme
def get_selection_scheme(validation, classification):
    if validation == 'random':
        return tpot2.selectors.random_selector
    elif (validation == 'unaggregated' or validation == 'aggregated') and classification:
        return lexicase_selection_no_comp
    elif (validation == 'unaggregated' or validation == 'aggregated') and not classification:
        # todo: add lexicase selection for regression (ask Anil for his implementation)
        return tpot2.selectors.lexicase_selection_regression
    elif validation == 'compressed':
        return tpot2.selectors.tournament_selection
    else:
        raise ValueError(f"Unknown validation: {validation}")

# pipeline search space: selector(optional) -> transformer(optional) -> selector(optional) -> regressor/classifier(mandatory)
def get_pipeline_space(classification, seed):
    if classification:
        return tpot2.search_spaces.pipelines.SequentialPipeline([
            tpot2.config.get_search_space(["selectors_classification","Passthrough"], random_state=seed),
            tpot2.config.get_search_space(["transformers","Passthrough"], random_state=seed),
            tpot2.config.get_search_space(["selectors_classification","Passthrough"], random_state=seed),
            tpot2.config.get_search_space("classifiers", random_state=seed)])
    else:
        return tpot2.search_spaces.pipelines.SequentialPipeline([
            tpot2.config.get_search_space(["selectors_regression","Passthrough"], random_state=seed),
            tpot2.config.get_search_space(["transformers","Passthrough"], random_state=seed),
            tpot2.config.get_search_space(["selectors_regression","Passthrough"], random_state=seed),
            tpot2.config.get_search_space("regressors", random_state=seed)])

# get estimator parameters depending on the selection scheme
def get_estimator_params(n_jobs,
                         classification,
                         validation,
                         X_train,
                         y_train,
                         seed):
    if classification:
        cv = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    else:
        cv = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    print('CV:', cv)

    # negative weights for regression tasks and positive weights for classification tasks
    objective_weights = -1.0 if not classification else 1.0

    # get selection objective functions
    if validation == 'unaggregated':
        # create selection objective functions
        objective_scorer = partial(unaggregated_selection_objectives,
                                   X=X_train,
                                   y=y_train,
                                   cv=cv,
                                   classification=classification)
        objective_scorer.__name__ = 'unaggregated-objectives'
        # create list of objective names per sample in y_select + complexity
        objective_names = ['obj_'+str(i) for i in range(y_train.shape[0])] + ['complexity']
        objective_weights = [objective_weights for _ in range(y_train.shape[0])] + [-1.0]

    elif validation == 'aggregated':
        # create selection objective functions
        objective_scorer = partial(aggregated_selection_objectives,
                                   X=X_train,
                                   y=y_train,
                                   cv=cv,
                                   classification=classification)
        objective_scorer.__name__ = 'aggregated-objectives'
        # create list of objective names per sample in y_select + complexity
        objective_names = ['fold_'+str(i) for i in range(10)] + ['complexity']
        objective_weights = [objective_weights for _ in range(10)] + [-1.0]

    elif validation == 'compressed'  or validation == 'random':
        # create selection objective functions
        objective_scorer = partial(compressed_selection_objectives,
                                   X=X_train,
                                   y=y_train,
                                   cv=cv,
                                   classification=classification)
        objective_scorer.__name__ = 'compressed-objectives'
        # create list of objective names per sample in y_select + complexity
        objective_names = ['cv'] + ['complexity']
        objective_weights = [objective_weights, -1.0]

    else:
        raise ValueError(f"Unknown validation: {validation}")

    return objective_names, cv, {
        # evaluation criteria
        'scorers': [],
        'scorers_weights':[],
        'cv': sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=seed), # not used
        'other_objective_functions': [objective_scorer],
        'other_objective_functions_weights': objective_weights,
        'objective_function_names': objective_names,

        # evolutionary algorithm params
        'population_size' : 10,
        'generations' : 3,
        'n_jobs':n_jobs,
        'survival_selector' :None,
        'parent_selector': get_selection_scheme(validation, classification),
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
        'verbose':1,
        'max_eval_time_seconds':60*5,
        'max_time_seconds': float("inf"),

        # pipeline search space
        'search_space': get_pipeline_space(classification, seed),
        }

# get test scores
def score(est, X, y, X_train, y_train, classification):
    # train evovled pipeline on the training data
    est.fit(X_train, y_train)

    # calculate testing performance
    if classification:
        # get classification testing score
        performance = np.float32(sklearn.metrics.get_scorer("accuracy")(est, X, y))
    else:
        # regression testing score
        performance = 0
    return {'testing_performance': performance}

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

# get the best pipeline from tpot2 depending on the selection scheme
def get_best_pipeline_results(est, cv_splits, validation, seed):
    sub = None
    # produce the 10 fold cross validation scores from aggregated selection objectives
    if validation == 'unaggregated':
        # remove rows with NaN values
        sub = est.evaluated_individuals.dropna(subset=['obj_0'])

        # get scores for each fold
        for f, (_, test_index) in enumerate(cv_splits):
            # make sure f is bettween 0 and 9
            assert f >= 0 and f < 10

            # create objectives names for each fold
            fold_names = ['obj_'+str(i) for i in test_index]

            # average all scores for a fold and add it as a new column
            sub['fold_'+str(f)] = sub[fold_names].mean(axis=1)

    # subset sub to only include the columns we are interested in
    # columns being the 'Inividual' and the 'fold' columns
    sub = sub[['Individual'] + [f'fold_{i}' for i in range(10)] + ['complexity']]

    # produce a single cross validation score from aggregated selection objectives
    if validation == 'aggregated' or validation == 'unaggregated':
        # calculate the mean of all the fold scores
        sub['cv'] = sub[[f'fold_{i}' for i in range(10)]].mean(axis=1)

    # get pipelines with the best cv score
    best_cv_pipelines = sub[sub['cv'] == sub['cv'].max()]
    # get pipelines with smallest complexity
    best_complexity_pipelines = best_cv_pipelines[best_cv_pipelines['complexity'] == best_cv_pipelines['complexity'].min()]
    # randomly select a pipeline from the best pipelines
    best_performer = best_complexity_pipelines.sample(n=1, random_state=seed)

    # return performance, complexity, and individual
    return np.float32(best_performer['cv'].values[0]), np.int64(best_performer['complexity'].values[0]), best_performer['Individual'].values[0].export_pipeline()

# execute task with tpot2
def execute_experiment(validation, task_id, n_jobs, save_path, seed, classification):
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

        # print data sizes
        print('X_train:', X_train.shape)
        print('y_train:', y_train.shape)

        # get estimator parameters
        names, cv, est_params = get_estimator_params(n_jobs=n_jobs,classification=classification,validation=validation, X_train=X_train,y_train=y_train,seed=seed)

        est = tpot2.TPOTEstimator(**est_params)

        start = time.time()
        print("ESTIMATOR FITTING")
        est.fit(X_train, y_train)
        duration = time.time() - start
        print("ESTIMATOR FITTING COMPLETE:", duration / 60 / 60, 'hours')

        # get best performer performance and cast to numpy float32d
        train_performance, complexity, pipeline = get_best_pipeline_results(est, cv.split(X_train, y_train),validation, seed)
        results = score(pipeline, X_test, y_test, X_train=X_train, y_train=y_train, classification=classification)
        results['training_performance'] = train_performance
        results['complexity'] = complexity
        results["task_id"] = task_id
        results["validation"] = validation
        results["seed"] = seed

        print('RESULTS:', results)

        print('SAVING:SCORES.PKL')
        with open(f"{save_folder}/results.pkl", "wb") as f:
            pickle.dump(results, f)

    except Exception as e:
        trace =  traceback.format_exc()
        pipeline_failure_dict = {"task_id": task_id, "validation": validation, "seed": seed, "error": str(e), "trace": trace}
        print("failed on ")
        print(save_folder)
        print(e)
        print(trace)

        with open(f"{save_folder}/failed.pkl", "wb") as f:
            pickle.dump(pipeline_failure_dict, f)

    return