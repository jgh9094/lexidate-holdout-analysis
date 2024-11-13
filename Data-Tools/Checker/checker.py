# outputs a list of directories that are incomplete

import os
import pickle

selection_scheme = {'Lexicase':0, 'Tournament':3000}

experiment_replicate_offset = 10000
seed_offsets = {'95':0,
                '90':500,
                '50':1000,
                '10':1500,
                '05':2000}

split_dirs = {'95':'learn_05_select_95',
              '90':'learn_10_select_90',
              '50':'learn_50_select_50',
              '10':'learn_90_select_10',
              '05':'learn_95_select_05'}

lex_unfinished_dirs = {'learn_05_select_95':[],
              'learn_10_select_90':[],
              'learn_50_select_50':[],
              'learn_90_select_10':[],
              'learn_95_select_05':[]}

tor_unfinished_dirs = {'learn_05_select_95':[],
              'learn_10_select_90':[],
              'learn_50_select_50':[],
              'learn_90_select_10':[],
              'learn_95_select_05':[]}

keys = ['95', '90', '50', '10', '05']

data_keys = ['testing_performance' ,'testing_complexity' ,'training_performance' ,'training_complexity' ,'task_id' ,'selection' ,'seed', 'split']

classification_tasks = [146818,359954,359955,190146,168757,359956,359958,359959,2073,359960,168784,359962]
openml_tasks = classification_tasks

reps = 40
data_dir = '/home/hernandezj45/Repos/lexidate-variation-analysis/Results/'

# check if data was successfully collected
def check_data_dir(dir):
    # check if the directory exists
    if os.path.isdir(data_dir + dir) == False:
        print('os.path.isdir(data_dir + dir) == False:', dir)
        return False

    # check if the data was collected
    if os.path.isfile(f'{data_dir}/{dir}/results.pkl') == False:
        print('os.path.isfile(f\'{data_dir}/{dir}/results.pkl\') == False:', dir)
        return False

    # open the pkl file
    results = pickle.load(open(f'{data_dir}/{dir}/results.pkl', 'rb'))

    # make sure results is of type dict
    if type(results) != dict:
        print('type(results) != dict:', dir)
        return False

    # check if the keys are present
    for key in data_keys:
        if key not in results.keys():
            print('key not in results.keys():', key '---', dir)
            return False

    return True

# generate a list of directories to pull data from
def go_though_all_dirs():
    # go through each selection scheme, key, and task
    for scheme, scheme_offset in selection_scheme.items():
        for key in keys:
            for i, task in enumerate(openml_tasks):
                for rep in range(1,reps+1):
                    # calcualte seed and check dir
                    seed = rep + (reps * i)
                    curr = f'{scheme}/{split_dirs[key]}/{scheme_offset + seed_offsets[key] + seed + experiment_replicate_offset}-{task}'
                    print('checking:', curr)

                    # check if the data was collected
                    if check_data_dir(curr) == False:
                        # add to the list of directories to check
                        if scheme == 'Lexicase':
                            lex_unfinished_dirs[split_dirs[key]].append(seed)
                        elif scheme == 'Tournament':
                            tor_unfinished_dirs[split_dirs[key]].append(seed)
                        else:
                            exit('Scheme not found')
    return

def main():
    go_though_all_dirs()

    # print out the directories that are incomplete
    print('*'*100)
    print('Lexicase')
    for key, val in lex_unfinished_dirs.items():
        output = ",".join(map(str, val))
        print(f'{key}: {output}')
        print('total:', len(val))
        print()
    print('*'*50)
    print('Tournament')
    for key, val in tor_unfinished_dirs.items():
        output = ",".join(map(str, val))
        print(f'{key}: {output}')
        print('total:', len(val))
        print()
    print('*'*100)


if __name__ == "__main__":
    main()