from cmind import utils
import os


def preprocess(i):

    env = i['env']

    script_path = i['run_script_input']['path']

    input('xyz11')


    return {'return': 0}

def postprocess(i):
    env = i['env']

    input('xyz22')
    
    return {'return': 1, 'error':'TBD'} #todo

    
    if env.get('CM_DATASET_CALIBRATION','') == "no":
        env['CM_DATASET_PATH_ROOT'] = os.path.join(os.getcwd(), 'install')
        env['CM_DATASET_PATH'] = os.path.join(os.getcwd(), 'install', 'validation', 'data')
        env['CM_DATASET_CAPTIONS_DIR_PATH'] = os.path.join(os.getcwd(), 'install', 'captions')
        env['CM_DATASET_LATENTS_DIR_PATH'] = os.path.join(os.getcwd(), 'install', 'latents')
    else:
        env['CM_CALIBRATION_DATASET_PATH'] = os.path.join(os.getcwd(), 'install', 'calibration', 'data')

    return {'return': 0}
