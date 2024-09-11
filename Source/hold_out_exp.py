import argparse
import Source.hold_out_utils as hold_out_utils

# to run: rm -r ./0-167184; python main_hold_out.py -split 0.5 -scheme tournament -task_id 167184 -n_jobs 10 -savepath ./ -seed 0 -task_type 1

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
    task_type = bool(args.task_type)
    # if task_type is True, then we are doing classification
    if task_type:
        print('Task Type: Classification')
    else:
        print('Task Type: Regression')

    # the test tasks used by autosklearn2 paper
    # task_id_lists = [ 189865,167200,126026,189860,75127,189862,75105,168798,126029,168796,
    #                  167190,167104,167083,167184,126025,75097,167181,168797,189861,167161,167149
    #                 ]

    # subset of tasks used for preliminary experiments
    task_id_lists = [167184,167181,167161,126026]
    assert task_id in task_id_lists, 'Task ID not in list of tasks'

    # execute task
    hold_out_utils.execute_experiment(split_select, scheme, task_id, n_jobs, save_path, seed, task_type)



if __name__ == '__main__':
    main()
    print('FINISHED')