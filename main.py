import sys

from sklearn.metrics import average_precision_score

from GDCMAD.data_utils import *
from GDCMAD.eval_methods import bf_search
from GDCMAD.options import Options
from GDCMAD.model import *
from torch.utils.data import DataLoader
import os

opt = Options().parse()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batchsize = 128

opt.train_batchsize = batchsize
opt.val_batchsize = batchsize
opt.test_batchsize = batchsize

opt.niter = 250


def anomaly_detection(data_type):
    #
    file_name_1 = "model"

    with open(file_name_1 + '.txt', 'w') as f:

        f.write('channel' + '\t' + 'dataset' + '\t' + 'f1' + '\t' + 'pre' + '\t' + 'rec' + '\t' +
                'tp' + '\t' + 'tn' + '\t' + 'fp' + '\t' + 'fn' + '\n')

        opt.dataset = data_type

        if data_type == 'SMAP':
            opt.dim = 25

        elif data_type == 'MSL':
            opt.dim = 55

        elif data_type == 'SMD':
            opt.dim = 38

        elif data_type == 'SWAT':
            opt.dim = 51
        elif data_type == 'PSM':
            opt.dim = 25
        elif data_type == 'ASD':
            opt.dim = 19
        else:
            print("There is no this dataset!!!!")

        path_train = os.path.join(os.getcwd(), "datasets", "train", data_type)
        files = os.listdir(path_train)

        for file in files:
            opt.filename = file
            data_name = data_type + '/' + str(file)
            print('file:', data_name)

            f_name = data_name.split('/')[1].split('.')[0]

            #
            file_name_2 = "model"

            label_name = open('label_' + file_name_2 + '.txt', 'w')

            samples_train_data, samples_val_data = read_train_data(opt.window_size, file=data_name,
                                                                   step=opt.step)
            print('train samples', samples_train_data.shape)
            print('valid samples', samples_val_data.shape)
            train_data = DataLoader(dataset=samples_train_data, batch_size=opt.train_batchsize, shuffle=True)
            #
            val_data = DataLoader(dataset=samples_val_data, batch_size=opt.val_batchsize, shuffle=True)
            #
            samples_test_data, test_label = read_test_data(opt.window_size, file=data_name)
            print('test samples', samples_test_data.shape)

            test_data = DataLoader(dataset=samples_test_data, batch_size=opt.test_batchsize)

            model = GDCMADModel(opt)
            model = to_device(model, device)

            model_path = "./train_model" + "/" + file_name_2 + ".pth"

            train_time, history, epoch_time = training(opt, model, train_data, val_data, model_path)
            model.load_state_dict(torch.load(model_path))

            #
            results, test_time = testing(model, test_data)

            # ()
            windows_labels = []
            for i in range(len(test_label) - opt.window_size):
                windows_labels.append(list(np.int_(test_label[i:i + opt.window_size])))

            y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]
            y_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                     results[-1].flatten().detach().cpu().numpy()])
            #
            y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))

            if len(y_test) >= len(y_pred):
                y_test = y_test[:len(y_pred)]
            else:
                y_pred = y_pred[:len(y_test)]
            #

            #
            t, th, label, point_label = bf_search(y_pred, y_test)

            for i in range(len(label)):
                label_name.write(str(label[i]) + '\n')
            label_name.close()

            print(str(data_type) + '\t' + str(file) + '\tf1=' + str(t[0]) + '\tpre=' + str(t[1]) +
                  '\trec=' + str(t[2]) + '\ttp=' + str(t[3]) + '\ttn=' + str(t[4]) + '\tfp=' + str(t[5]) +
                  '\tfn=' + str(t[6]))

            f.write(str(data_type) + '\t' + str(file) + '\t' + str(t[0]) + '\t' + str(t[1]) + '\t' +
                    str(t[2]) + '\t' + str(t[3]) + '\t' + str(t[4]) + '\t' + str(t[5]) + '\t' + str(t[6]) +
                    '\t' + str(train_time) + '\t' + str(epoch_time) + '\t' + str(test_time) + '\t' + str(
                th) + '\t' + '\n')

    print('finished')


if __name__ == '__main__':
    commands = sys.argv[2]

    anomaly_detection(commands)
