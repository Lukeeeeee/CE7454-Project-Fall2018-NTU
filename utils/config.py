import numpy as np
import os
from data import DATA_PATH
from log import LOG_PATH
from data.list import LIST_PATH
import time
import json

SRC_PATH=os.path.abspath('..')
MODEL_PATH=os.path.abspath('../model')
class Config(object):
    # Setting dataset directory

    

    Kaggle_DATA_DIR = os.path.join(DATA_PATH)
    Kaggle_eval_list = os.path.join(LIST_PATH, 'valid.txt')
    Kaggle_train_list = os.path.join(LIST_PATH, 'train.txt')

    # B G R order
    IMG_MEAN = np.array((177.682, 175.84, 174.21), dtype=np.float32)


    model_paths = {'train': os.path.join(MODEL_PATH, 'icnet_cityscapes_train_30k_bnnomerge.npy'),
                   'trainval': os.path.join(MODEL_PATH,  'icnet_cityscapes_train_30k_bnnomerge.npy'),
                   'train_bn': os.path.join(MODEL_PATH, 'icnet_cityscapes_train_30k_bnnomerge.npy'),
                   'trainval_bn': os.path.join(MODEL_PATH,  'icnet_cityscapes_train_30k_bnnomerge.npy'),
                   'others': os.path.join(LOG_PATH, '2018-11-03_14-04-58/model.ckpt-4999'),
                   }
    ## If you want to train on your own dataset, try to set these parameters.
    others_param = {'name': 'Kaggle',
                    'num_classes': 2,
                    'ignore_label': 100,
                    'eval_size': [1280, 1918],
                    'eval_steps': 1024,
                    'eval_list': Kaggle_eval_list,
                    'train_list': Kaggle_train_list,
                    'loss_type': 'Cross Entropy Loss',
                    'total_train_sample': 4064,
                    'total_eval_sample': 1024,
                    'data_dir': Kaggle_DATA_DIR}

    ## You can modify following lines to train different training configurations.

    TRAINING_SIZE = [720, 720]

    # previously 60001
    TRAINING_EPOCHS = 20

    N_WORKERS = 8
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9
    POWER = 0.9
    RANDOM_SEED = int(round(time.time() * 1000) % 2 ** 31 - 1)
    WEIGHT_DECAY = 0.0001

    SAVE_NUM_IMAGES = 4
    SAVE_PRED_EVERY = 2

    # Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
    LAMBDA1 = 0.16
    LAMBDA2 = 0.4
    LAMBDA3 = 1.0

    def __init__(self, dataset, is_training=False, filter_scale=1, random_scale=False, random_mirror=False,
                 log_path_end='', eval_path_log=None,INFER_SIZE=None):
        print('Setup configurations...')
        if eval_path_log:
            self.SNAPSHOT_DIR = eval_path_log
        else:
            self.SNAPSHOT_DIR = os.path.join(LOG_PATH, time.strftime("%Y-%m-%d_%H-%M-%S") + '_' + log_path_end)
            while os.path.exists(self.SNAPSHOT_DIR):
                self.SNAPSHOT_DIR = os.path.join(LOG_PATH, time.strftime("%Y-%m-%d_%H-%M-%S") + '_' + log_path_end)
            os.mkdir(self.SNAPSHOT_DIR)
        if dataset == 'ade20k':
            self.param = self.ADE20k_param
        elif dataset == 'cityscapes':
            self.param = self.cityscapes_param
        elif dataset == 'others':
            self.param = self.others_param

        self.dataset = dataset
        self.random_scale = random_scale
        self.random_mirror = random_mirror
        self.is_training = is_training
        self.filter_scale = filter_scale
        self.INFER_SIZE=INFER_SIZE
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        log_config = {}
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not isinstance(getattr(self, a), dict):
                print("{:30} {}".format(a, getattr(self, a)))
                if isinstance(getattr(self, a), np.ndarray):
                    log_config[a] = getattr(self, a).tolist()
                elif isinstance(getattr(self, a), np.float32):
                    log_config[a] = float(getattr(self, a))
                else:
                    log_config[a] = getattr(self, a)

            if a == ("param"):
                print(a)
                for k, v in getattr(self, a).items():
                    print("   {:27} {}".format(k, v))
                    log_config[k] = v

        print("\n")
        self.save_to_json(dict=log_config, path=os.path.join(self.SNAPSHOT_DIR, 'config.json'),mode='eval')

    @staticmethod
    def load_json(file_path):
        with open(file_path, 'r') as f:
            res = json.load(f)
            return res

    @staticmethod
    def save_to_json(mode,dict, path, file_name=None):
        if file_name is not None:
            path = os.path.join(path, file_name)
        if mode=='eval' or mode=='inference':

            with open(path, 'w') as f:
                json.dump(obj=dict, fp=f, indent=4, sort_keys=True)
        else:
            fr = open(path)
            model = json.load(fr)
            fr.close()
            for i in dict:
                model[i] = dict[i]

            jsObj = json.dumps(model)

            with open(path, "w") as fw:
                fw.write(jsObj)
                fw.close()
