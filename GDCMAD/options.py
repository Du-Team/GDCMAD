import argparse
import os
import torch


class Options:

    def __init__(self):

        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('--seed', default=12345, help='random seed')
        self.parser.add_argument('--dataset', default='SMD', help='dataset')
        self.parser.add_argument('--filename', default='machine-1-1.txt', help='dataset name')
        self.parser.add_argument('--train_batchsize', type=int, default=50, help='batch size of train data')
        self.parser.add_argument('--val_batchsize', type=int, default=50, help='batch size of validation data')
        self.parser.add_argument('--test_batchsize', type=int, default=50, help='batch size of test data')
        self.parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
        self.parser.add_argument('--step', type=int, default=1, help='sequence step')  
        self.parser.add_argument('--window_size', type=int, default=5, help='sequence length')
        self.parser.add_argument('--dim', type=int, default=38, help='dimensions of data')
        self.parser.add_argument('--project_channels', type=int, default=32, help='channels of project')
        self.parser.add_argument('--lstm_h_dim', type=int, default=32)
        self.parser.add_argument('--drop_out', type=int, default=0.2, help='')
        self.parser.add_argument('--kernel_size', type=int, default=2, help='')
        self.parser.add_argument('--loss_cl_ratio', type=float, default=1, help='')
        self.parser.add_argument('--loss_cl_enc_ratio', type=float, default=1, help='')
        self.parser.add_argument('--device', type=str, default='gpu', help='Device: gpu | cpu')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0. use -1 for CPU')
        self.parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
        self.parser.add_argument('--model', type=str, default='GDCMADModel', help='detection model')
        self.parser.add_argument('--outf', default='./output', help='output folder')

        ##
        # Train
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--niter', type=int, default=250, help='number of epochs to train for')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--patience', type=int, default=5, help='early stopping')
        self.parser.add_argument('--delta', type=float, default=0.05, help='early stopping delta')

        ## Test
        self.parser.add_argument('--istest', action='store_true', help='train model or test model')
        
        self.opt = None

    def parse(self):

        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if self.opt.device == 'gpu':
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)  

        self.opt.name = "%s/%s" % (self.opt.model, self.opt.dataset)
        expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
