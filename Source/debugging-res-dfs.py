from hold_out_utils import execute_experiment
import pandas as pd
import math

pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Parameters
split_select = 0.25
n_jobs = 10
savepath = './'
seed = 1

schemes = ['random','tournament','lexicase']
task_id_type = [(359934, False),(146818, True)]

# function to make sure that df1 and df2 are the same
def compare_dfs(df, df2):
    # make sure they have the same length
    if len(df) != len(df2):
        print('len(df) != len(df2)')
        print('len(df):', len(df))
        print('len(df2):', len(df2))
        return False

    # iterate through both dfs and compare the 'Instance' column to make sure that the results are the same
    for i in range(len(df)):

        # go through each step in the pipeline and compare the parameters
        for step1, step2 in zip(df.iloc[i]['Instance'], df2.iloc[i]['Instance']):
            # make sure thy are the same type
            if type(step1) != type(step2):
                print('i:', i)
                print('type(step1) != type(step2)')
                print('step1:', step1)
                print('step2:', step2)
                print('step1 type:', type(step1))
                print('step2 type:', type(step2))
                return False

            # make sure they have the same keys
            if step1.get_params().keys() != step2.get_params().keys():
                print('i:', i)
                print('same-keys')
                print('step1 keys:\n', step1.get_params().keys())
                print('type:', type(step1))
                print('step2 keys:\n', step2.get_params().keys())
                print('type:', type(step2))
                return False

            # make sure they have the same parameters
            for key in step1.get_params().keys():

                # custom check for classifer and regressor
                if key == 'estimator':
                    # make sure its not None
                    if step1.get_params()[key] == None and step2.get_params()[key] == None:
                        continue

                    # make sure they ar the same type
                    if type(step1.get_params()[key]) != type(step2.get_params()[key]):
                        print('i:', i)
                        print('estimator-type:', key)
                        print('step1:', type(step1.get_params()[key]))
                        print('step2:', type(step2.get_params()[key]))
                        return False

                    # make sure they have the same parameters
                    if step1.get_params()[key].get_params() != step2.get_params()[key].get_params():
                        print('i:', i)
                        print('estimator-param:', key)
                        print('step1 keys:', step1.get_params()[key].get_params())
                        print('step2 keys:', step2.get_params()[key].get_params())
                        return False

                # else check the parameters
                else:
                    if isinstance(step1.get_params()[key], float) and isinstance(step2.get_params()[key], float):
                        # check if both are nan
                        if math.isnan(step1.get_params()[key]) and math.isnan(step2.get_params()[key]):
                            continue

                    if step1.get_params()[key] != step2.get_params()[key]:
                        print('i:', i)
                        print('key-check')
                        print('key:', key)
                        print('step1:', step1.get_params()[key])
                        print('step2:', step2.get_params()[key])
                        return False


        # go through each 'Parents' column and compare them
        if isinstance(df.iloc[i]['Parents'], tuple) or isinstance(df2.iloc[i]['Parents'], tuple):
            if df.iloc[i]['Parents'] != df2.iloc[i]['Parents']:
                print('i:', i)
                print('Parents')
                print('p1:', df.iloc[i]['Parents'])
                print('type:', type(df.iloc[i]['Parents']))
                print('p2:', df2.iloc[i]['Parents'])
                print('type:', type(df2.iloc[i]['Parents']))
                return False

        # go through each 'Variation_Function' column and compare them
        if isinstance(df.iloc[i]['Variation_Function'], str) or isinstance(df2.iloc[i]['Variation_Function'], str):
            if df.iloc[i]['Variation_Function'] != df2.iloc[i]['Variation_Function']:
                print('i:', i)
                print('Variation_Function')
                print('p1:', df.iloc[i]['Variation_Function'])
                print('p2:', df2.iloc[i]['Variation_Function'])
                return False

        # go through each 'Eval Error' column and compare them
        if df.iloc[i]['Eval Error'] != df2.iloc[i]['Eval Error']:
            print('i:', i)
            print('Eval Error')
            print('p1:', df.iloc[i]['Eval Error'])
            print('p2:', df2.iloc[i]['Eval Error'])
            return False

        # go through each 'Pareto_Front' column and compare them
        if math.isnan(df.iloc[i]['Pareto_Front']) == False or math.isnan(df2.iloc[i]['Pareto_Front']) == False:
            if df.iloc[i]['Pareto_Front'] != df2.iloc[i]['Pareto_Front']:
                print('i:', i)
                print('Pareto_Front')
                print('p1:', df.iloc[i]['Pareto_Front'])
                print('p2:', df2.iloc[i]['Pareto_Front'])
                return False

        # go though each performance metric and compare them
        if math.isnan(df.iloc[i]['performance']) == False or math.isnan(df2.iloc[i]['performance']) == False:
            if df.iloc[i]['performance'] != df2.iloc[i]['performance']:
                print('i:', i)
                print('performance')
                print('p1:', df.iloc[i]['performance'])
                print('p2:', df2.iloc[i]['performance'])
                return False

        # go though each complexity metric and compare them
        if math.isnan(df.iloc[i]['complexity']) == False or math.isnan(df2.iloc[i]['complexity']) == False:
            if df.iloc[i]['complexity'] !=  df2.iloc[i]['complexity']:
                print('i:', i)
                print('complexity')
                print('p1:', df.iloc[i]['complexity'])
                print('p2:', df2.iloc[i]['complexity'])
                return False
    return True

# Will compare the results and dataframes from two different experiments with
# the same parameters to make sure that the results are the same and reproducible.
# Assuming that 'execute_experiment' is a function that returns the estimator, results, and dataframe
def main():
    # iterate through selected schemes
    for scheme in schemes:
        print('Scheme:', scheme)
        for task_id, task_type in task_id_type:
            print('Task ID:', task_id)
            # create the first estimator and get the results and dataframe
            est1, res1, df = execute_experiment(split_select, scheme, task_id, n_jobs, savepath, seed, task_type)
            # create new estimator with the same parameters and make sure that the results are the same
            for _ in range(10):
                print('Iteration:', _)
                # create the second estimator and get the results and dataframe
                est2, res2, df2  = execute_experiment(split_select, scheme, task_id, n_jobs, savepath, seed, task_type)

                assert res1 == res2
                assert compare_dfs(df, df2)

if __name__ == '__main__':
    main()
    print('FINISHED')