from src.evaluate import main as eval
import os
import tensorflow as tf

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    path_list = [
        '/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/log/2018-11-07_19-37-16__v2_DEFAULT_CONFIG_LAMBDA_0.160000_0.400000_1.000000',
        '/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/log/2018-11-08_03-10-15__v2_DEFAULT_CONFIG_LAMBDA_0.120000_0.400000_1.000000',
        '/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/log/2018-11-08_01-14-34__v2_DEFAULT_CONFIG_LAMBDA_0.200000_0.400000_1.000000',
        '/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/log/2018-11-08_10-52-15__v2_DEFAULT_CONFIG_LR_0.000100',
        '/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/log/2018-11-08_05-05-45__v2_DEFAULT_CONFIG_LAMBDA_0.160000_0.400000_1.200000',
        '/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/log/2018-11-08_08-56-52__v2_DEFAULT_CONFIG_LR_0.000500',
        '/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/log/2018-11-08_12-41-35__v2_DEFAULT_CONFIG_LR_0.000500',
        '/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/log/2018-11-07_23-18-47__v2_DEFAULT_CONFIG_LAMBDA_0.160000_0.300000_1.000000',
        '/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/log/2018-11-08_07-00-01__v2_DEFAULT_CONFIG_LAMBDA_0.160000_0.400000_0.800000',
        '/home/dls/meng/DLProject/CE7454_Project_Fall2018_NTU/log/2018-11-07_21-27-30__v2_DEFAULT_CONFIG_LAMBDA_0.160000_0.500000_1.000000']
    for path in path_list:
        tf.reset_default_graph()
        eval(model_log_dir=path, check_point=19)
        sess = tf.get_default_session()
        if sess:
            sess._exit__(None, None, None)
