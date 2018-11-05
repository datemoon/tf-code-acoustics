
import sys
import os
import pickle
import time

import argparse 

class MyArgumentParser(argparse.ArgumentParser):
    '''
    This class inherit argparse.ArgumentParser.It's used analy command line.
    '''
    def convert_arg_line_to_args(self, arg_line):
        args_list = arg_line.replace(' ','').split('=')
        if args_list[0] == '' or args_list[0][0] == '#':
            return []
        ret_args = []
        num=0
        for arg in args_list:
            if num == 0:
                arg = arg.replace('_','-')
            if arg[0] == '#':
                break
            if '#' in arg:
                ret_args.append(arg.split('#')[0])
                break
            ret_args.append(arg)
            num += 1
        if len(ret_args) > 2:
            print('parameter ERROR:' + arg_line)
            raise 'config file parameter line have error'
        return ret_args

    def parse_args(self, args_line):
        self.conf_str_ = '--config'
        self.conf_file_ = ''
        self.fromfile_prefix_chars_ = '@'
        self.add_argument(self.conf_str_, type=str, default=self.conf_file_,
                help='Path to configuration file with hyper-parameters.'
                ' (default :  NULL )')
        self.args_ = []
        for args in args_line:
            args_list = self.convert_arg_line_to_args(args)
            if '--help' == args_list[0] or '-h' == args_list[0]:
                return argparse.ArgumentParser.parse_args(self, ['--help'])
            if self.conf_str_ ==  args_list[0]:
                if args_list[0] == self.conf_str_:
                    assert len(args_list) == 2
                    self.conf_file_ = args_list[1]
            self.args_.extend(args_list)
        if self.conf_file_ != '':
            self.args_.append(self.fromfile_prefix_chars_ + self.conf_file_)

        return argparse.ArgumentParser.parse_args(self,self.args_)

if __name__ == "__main__":
    with open('conf','w') as fp:
        fp.write('#config file\n\n--tf-save-path = outdir\n--left-context = 3\n--num-threads=6\n')
    parser = MyArgumentParser(description='this is a test code'
            'please input python parse_opt.py --config=conf',
            fromfile_prefix_chars='@')
    parser.add_argument('--tf-save-path', dest='tf_save_path',
            type=str, default=None,
            help='TensorFlow save path name for the run (allow multiples run with the same output path)')
    parser.add_argument('--num-threads', dest='num_threads', type=int, default=1,
            help='number threads (default : 1)')
    feature_opt = parser.add_argument_group(title='feature_opt',
            description='feature option relation parameters')
    feature_opt.add_argument('--left-context', dest='left_context',
            type=int, default=0,
            help='left context frame number'
            ' (int, default = 0)' )


    args = parser.parse_args(sys.argv[1:])
    print(args)
    print(args.__dict__)
