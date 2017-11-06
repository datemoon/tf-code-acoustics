
import argparse 



def parse_args(): 
    """ 
    Parses the command line input. 
 
    """ 
    _DEFAULT_CONFIG_FILE = 'conf/config.ini' 


    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default=_DEFAULT_CONFIG_FILE, 
                        help='Path to configuration file with hyper-parameters.'
                        ' (default : ' + _DEFAULT_CONFIG_FILE + ')') 
    parser.add_argument('--num_threads', type=int, default=1, help='number threads (default : 1)')
    parser.add_argument('--tf-save-path', dest='tf_save_path', 
            type=str, default=None, 
            help='TensorFlow save path name for the run (allow multiples run with the same output path)')

    parser.add_argument('--max-epoch', dest='max_epoch', type=int, default=None, 
                        help='Max epoch to train (no limitation if not provided)') 
    
    '''
        add feature option
    '''
    feature_opt = parser.add_argument_group(title='feature_opt', 
            description='feature option relation parameters')

    feature_opt.add_argument('--right-context', dest='right_context',
            type=int, default=0,
            help='right context frame number'
            ' (int, default = 0)')
    
    feature_opt.add_argument('--left-context', dest='left_context',
            type=int, default=0,
            help='left context frame number'
            ' (int, default = 0)')

    feature_opt.add_argument('--skip-frames', dest='skip_frames',
            type=int, default=0,
            help='skip frame number'
            ' (int, default = 0)')
    
    feature_opt.add_argument('--start-frames', dest='start_frames',
            type=int, default=0,
            help='start frame ,it must be lt skip-frames'
            ' (int, default = 0)')


    '''
       add train common option
    '''
    train_common_opt = parser.add_argument_group(title='train_common_opt', 
            description='training common option relation parameters')

    train_common_opt.add_argument('--learn-rate', dest='learn_rate',
            type=float, default=1.0, 
            help='learning rate for NN training'
            ' (float, default = 1.0)')

    train_common_opt.add_argument('--batch-size', dest='batch_size', type=int, 
            default=20,
            help='Length of \'one stream\' in the Multi-stream training'
            '(int, default = 20)')

    train_common_opt.set_defaults(cross_validate=False)
    train_common_opt.add_argument('--cross-validate', dest='cross_validate',
            action='store_true',
            help='Perform cross-validation (don\'t back-propagate)'
            ' (bool, default = false)')

    train_common_opt.add_argument('--momentum', type=float, default=0.0,
            help='Momentum' 
            ' (float, default = 0.0)')

    train_common_opt.add_argument('--objective-function', dest='objective_function',
            type=str, default='ctc',
            help='Objective function : ctc|xent|mse'
            '(string, default = "ctc")')
    
    train_common_opt.add_argument('--report-step', dest='report_step',type=int,
            default=100,
            help=' Step (number of sequences) for status reporting'
            '(int, default = 100)')


    '''
       add lstm train relation option
    '''
    train_lstm_opt = parser.add_argument_group(title='train_lstm_opt', 
            description='training lstm option relation parameters')
    train_lstm_opt.add_argument('--num-streams', dest='nstreams', type=int,
            default=1,
            help='Number of streams in the Multi-stream training'
            '(int, default = 4)')

    train_lstm_opt.add_argument('--frame-num-limit',dest='frame_num_limit',
            type=int, default=1500,
            help='Sentence max number of frames' 
            ' (double, default = 100000)')



    '''
       add mutually exclusive group 4 choise 1
    '''
    group = parser.add_mutually_exclusive_group(required=True) 
    group.set_defaults(train=False) 
    group.set_defaults(file=None) 
    group.set_defaults(record=False) 
    group.set_defaults(evaluate=False) 
    group.add_argument('--train', dest='train', action='store_true', help='Train the network') 
    group.add_argument('--file', type=str, help='Path to a wav file to process') 
    group.add_argument('--record', dest='record', action='store_true', help='Record and write result on the fly') 
    group.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate WER against the test_set') 

    '''
        dict choise
    '''
    args = parser.parse_args() 
    if args.start_frames >= args.skip_frames and args.start_frames != 0:
        raise 'error feature option'
    prog_params = {'config_file': args.config, 'tf_save_path': args.tf_save_path, 'num_threads':args.num_threads,
            'right_context': args.right_context, 'left_context': args.left_context, 'skip_frames': args.skip_frames, 'start_frames': args.start_frames,

            'max_epoch': args.max_epoch, 'learn_rate': args.learn_rate, 'batch_size': args.batch_size, 
            'cross_validate': args.cross_validate, 'momentum': args.momentum, 'objective_function': args.objective_function,
            'report_step': args.report_step, 
            
            'nstreams': args.nstreams, 'frame_num_limit': args.frame_num_limit, 
                   
            'train': args.train, 'file': args.file, 'record': args.record, 'evaluate': args.evaluate}
                   
    return prog_params 


if __name__ == "__main__":
    args = parse_args()
    print(args)



