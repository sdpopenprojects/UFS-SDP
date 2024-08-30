import matlab.engine
import scipy.io as sio
import numpy as np
import os

def getSub(train_data, test_data, UFS_NAME, current_dir):
    # Save train_data, train_label as a mat file and put it in the path of train_dataset_path
    sio.savemat(os.path.join(current_dir, 'temp_dataset.mat'), {'train_data': train_data, 'test_data': test_data})
    train_dataset_path = os.path.join(current_dir, 'temp_dataset.mat')

    # Starts a new MATLAB process and returns a Python variable that is a MatlabEngine object used to communicate with the MATLAB process.
    eng = matlab.engine.start_matlab()
    a = eng.UFS(train_dataset_path, UFS_NAME, nargout=3)
    eng.quit()

    train_data_sub = np.array(a[0])
    test_data_sub = np.array(a[1])
    order = np.array(a[2])

    return train_data_sub, test_data_sub, order