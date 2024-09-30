import os
import pickle
import pandas
import numpy as np

selection_scheme = ['Lexicase', 'Tournament', 'Random']
seed_offsets = {'75':0, '50':1000, '25':2000}
split_dirs = {'75':'train_25_test_75', '50':'train_50_test_50', '25':'train_75_test_25'}
keys = ['75', '50', '25']
data_keys = ['testing_performance' ,'testing_complexity' ,'training_performance' ,'training_complexity' ,'task_id' ,'selection' ,'seed']

regression_prelim_tasks = [359934, 359945, 359948, 359939]
classification_prelim_tasks = [146818, 168784, 190137, 359969]

reps = 20
data_dir = '/home/hernandezj45/Repos/lexidate-variation-analysis/Results/Holdout/'
collector = {'testing_performance':[],
             'testing_complexity': [],
             'training_performance': [],
             'training_complexity': [],
             'task_id': [],
             'selection': [],
             'seed': []}

# check if data was successfully collected
def check_data_dir(dirs):
    # incomplete data dirs
    incomplete_dirs = []

    # go through each dir and check if the data was collected
    for dir in dirs:
        # check if the directory exists
        if os.path.isdir(data_dir + dir) == False:
            incomplete_dirs.append(dir)
            continue

        # check if the data was collected
        if os.path.isfile(f'{data_dir}/{dir}/results.pkl') == False:
            incomplete_dirs.append(dir)
            continue

        # open the pkl file
        results = pickle.load(open(f'{data_dir}/{dir}/results.pkl', 'rb'))

        # make sure results is of type dict
        if type(results) != dict:
            incomplete_dirs.append(dir)
            continue

        # check if the keys are present
        for key in data_keys:
            if key not in results.keys():
                incomplete_dirs.append(dir)
                break

# generate a list of directories to pull data from
def generate_dirs():
    check_dirs = []
    for scheme in selection_scheme:
        print('scheme:', scheme)
        for key in keys:
            print('split:', key)
            for task in classification_prelim_tasks + regression_prelim_tasks:
                    for rep in range(1,reps+1):
                        check_dirs.append(f'{scheme}/{split_dirs[key]}/{seed_offsets[key]+rep}-{task}')
    return check_dirs


def main():
    todo = check_data_dir(generate_dirs())

    print('Incomplete Dirs:')
    for dir in todo:
        print(dir)

if __name__ == "__main__":
    main()