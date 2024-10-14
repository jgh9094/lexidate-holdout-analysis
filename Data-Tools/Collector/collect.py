# goes through each directory and grabs all data from the results.pkl file

import os
import sys
import pickle as pkl
import pandas as pd
from pathlib import Path

selection_scheme = ['Lexicase', 'Tournament'] #, 'Random']
split_dirs = {'90%':'learn_10_select_90',
              '80%':'learn_20_select_80',
              '50%':'learn_50_select_50',
              '20%':'learn_80_select_20',
              '10%':'learn_90_select_10'}
regression_tasks = [] # todo
classification_tasks = [359953,146818,359954,359955,190146,168757,359956,
                            359957,359958,359959,2073,10090,359960,168784,359961,
                            359962]

# assuming data is in the home directory
data_dir = str(Path.home()) + '/Repos/lexidate-variation-analysis/Results'
collector = {'testing_performance':[],
             'testing_complexity': [],
             'training_performance': [],
             'training_complexity': [],
             'task_id': [],
             'selection': [],
             'split': [],
             'task_type': [],
             'seed': []}

# check if data was successfully collected
def get_data():
    # go through each scheme data directory
    for scheme in selection_scheme:
        print('scheme:', scheme)
        # go through each split data directory
        for split, dir in split_dirs.items():
            exp_dir = f'{data_dir}/{scheme}/{dir}/'
            print('exp_dir:', exp_dir)

            # go though all subdirectories in the experiment directory
            for sub_dir, dirs, files in os.walk(exp_dir):
                # skip root dir
                if sub_dir == exp_dir:
                    continue

                print('sub_dir:', f'{sub_dir}/')

                # open the pkl file
                results = pkl.load(open(f'{sub_dir}/results.pkl', 'rb'))

                # add data to collector
                collector['testing_performance'].append(results['testing_performance'])
                collector['testing_complexity'].append(results['testing_complexity'])
                collector['training_performance'].append(results['training_performance'])
                collector['training_complexity'].append(results['training_complexity'])
                collector['task_id'].append(results['task_id'])
                collector['selection'].append(results['selection'])
                collector['split'].append(split)
                collector['seed'].append(results['seed'])
                if results['task_id'] in classification_tasks:
                    collector['task_type'].append('classification')
                elif results['task_id'] in regression_tasks:
                    collector['task_type'].append('regression')
                else:
                    exit(f'Error: task_id {results["task_id"]} not found in classification_tasks or regression_tasks')

    # create a dataframe from the collector dictionary and make a csv file
    df = pd.DataFrame(collector)
    df.to_csv('data.csv', index=False)


def main():
    get_data()

if __name__ == "__main__":
    main()