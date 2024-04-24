from cmind import utils
import os

def preprocess(i):
    os_info = i['os_info']

    env = i['env']

    state = i['state']

    print ('')
    print ('Current directory: {}'.format(os.getcwd()))

    print ('')

    extra = ''

    if env.get('CM_ABTF_NUM_CLASSES', '')!='':
        extra += ' --num-classes '+str(env['CM_ABTF_NUM_CLASSES'])

    if utils.check_if_true_yes_on(env, 'CM_USE_DATASET'):
        extra += ' --data-path ' + env['CM_DATASET_MLCOMMONS_COGNATA_PATH']

    if env.get('CM_INPUT_IMAGE', '')!='':
        extra += ' --input ' + env['CM_INPUT_IMAGE']

    if env.get('CM_OUTPUT_IMAGE', '')!='':
        extra += ' --output ' + env['CM_OUTPUT_IMAGE']

    if utils.check_if_true_yes_on(env, 'CM_ABTF_VISUALIZE'):
        extra += ' --visualize'

    if extra!='':
        print ('')
        print ('Extra command line: {}'.format(extra))

    env['CM_ABTF_EXTRA_CMD'] = extra

    return {'return':0}

def postprocess(i):

    return {'return':0}
