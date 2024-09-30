# goes through each directory and grabs all data from the results.pkl file

import os
import sys
import pickle as pkl
import pandas as pd
from pathlib import Path

selection_scheme = ['Lexicase', 'Tournament'] #, 'Random']
split_dirs = {'75%':'train_25_test_75', '50%':'train_50_test_50', '25%':'train_75_test_25'}

regression_prelim_tasks = [359934, 359945, 359948, 359933]
classification_prelim_tasks = [146818, 168784, 190137, 359969]

# assuming data is in the home directory
data_dir = str(Path.home()) + '/Repos/lexidate-variation-analysis/Results/Holdout'
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

                print('sub_dir:', f'{exp_dir}/{sub_dir}/')

                # open the pkl file
                results = pkl.load(open(f'{exp_dir}/{sub_dir}/results.pkl', 'rb'))

                # add data to collector
                collector['testing_performance'].append(results['testing_performance'])
                collector['testing_complexity'].append(results['testing_complexity'])
                collector['training_performance'].append(results['training_performance'])
                collector['training_complexity'].append(results['training_complexity'])
                collector['task_id'].append(results['task_id'])
                collector['selection'].append(results['selection'])
                collector['split'].append(split)
                collector['seed'].append(results['seed'])
                if results['task_id'] in classification_prelim_tasks:
                    collector['task_type'].append('classification')
                else:
                    collector['task_type'].append('regression')

    # create a dataframe from the collector dictionary and make a csv file
    df = pd.DataFrame(collector)
    df.to_csv('holdout_data.csv', index=False)


def main():
    get_data()

if __name__ == "__main__":
    main()