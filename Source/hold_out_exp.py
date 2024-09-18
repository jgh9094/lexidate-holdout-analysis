import argparse
import hold_out_utils as hold_out_utils

# to run: rm -r ./0-146818; python main_hold_out.py -split 0.5 -scheme tournament -task_id 146818 -n_jobs 10 -savepath ./ -seed 0 -task_type 1

def main():
    # read in arguements
    parser = argparse.ArgumentParser()
    # split proportion we are using
    parser.add_argument("-split_select", required=True, nargs='?')
    # what selection scheme are we using
    parser.add_argument("-scheme", required=True, nargs='?')
    # what openml task are we using
    parser.add_argument("-task_id", required=True, nargs='?')
    # number of threads to use during estimator evalutation
    parser.add_argument("-n_jobs",  required=True, nargs='?')
    # where to save the results/models
    parser.add_argument("-savepath", required=True, nargs='?')
    # seed offset
    parser.add_argument("-seed", required=True, nargs='?')
    # is this a classification (True) or regression (False) task
    parser.add_argument("-task_type", required=True, nargs='?')

    args = parser.parse_args()
    split_select = float(args.split_select)
    print('Split:', split_select)
    scheme = str(args.scheme)
    print('Scheme:', scheme)
    task_id = int(args.task_id)
    print('Task ID:', task_id)
    n_jobs = int(args.n_jobs)
    print('Number of Jobs:', n_jobs)
    save_path = str(args.savepath)
    print('Save Path:', save_path)
    seed = int(args.seed)
    print('Seed:', seed)
    task_type = bool(int(args.task_type))
    # if task_type is True, then we are doing classification
    if task_type:
        print('Task Type: Classification')
    else:
        print('Task Type: Regression')

    # keep this in mind for the preliminary tasks
    regression_prelim_tasks = [359934, 359945, 359948, 359939]

    # keep this in mind for the preliminary tasks
    classification_prelim_tasks = [146818, 168784, 190137, 359969]

    assert task_id in regression_prelim_tasks + classification_prelim_tasks, 'Task ID not in list of tasks'

    # execute task
    hold_out_utils.execute_experiment(split_select,
                                      scheme,
                                      task_id,
                                      n_jobs,
                                      save_path,
                                      seed,
                                      task_type)

if __name__ == '__main__':
    main()
    print('FINISHED')