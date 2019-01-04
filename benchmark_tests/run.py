'''
Usage:
Here, we use resnet as an example.
1. # python3 run.py resnet --dump
    * Modify config_resnet to fit your environment and requirement.
2. # python3 run.py @config_resnet
'''
import argparse
import csv
import os
import pprint
import re
import signal
import subprocess
import sys
import time



def sigint_handler(signum, frame):
    print('Ctrl + C has been pressed.')
    print('Killing the subprocess.')
    #os.killpg(os.getpgid(sigint_handler.proc.pid), signal.SIGINT)
    sigint_handler.proc.send_signal(signal.SIGINT); 
    time.sleep(1)
    print('Proc killed')
    #sys.exit(0)


def register_sigint(proc):
    sigint_handler.proc = proc
    signal.signal(signal.SIGINT, sigint_handler)


def run_script(args, envs):
    if args.model == 'resnet':
        log_name = './log_{}_{}_{}_{}_{}_{}_{}'.format(
                        args.model, args.num_layers, args.batch_size,
                        args.wide_scale, envs['MXNET_SWAP_ALGORITHM'],
                        envs['MXNET_PREFETCH_ALGORITHM'],
                        envs['MXNET_PREFETCH_STEP_AHEAD'])
        options = ['--num_gpus={}'.format(args.num_gpus),
                   '--num_loops={}'.format(args.num_loops),
                   '--num_layers={}'.format(args.num_layers),
                   '--wide_scale={}'.format(args.wide_scale),
                   '--batch_size={}'.format(args.batch_size)]
    elif args.model == 'rnn':
        log_name = 'TBD'
        options = []
        raise NotImplementedError
    elif args.model == 'inception-v4':
        log_name = './log_{}_{}_{}_{}_{}'.format(
                        args.model, args.batch_size,
                        envs['MXNET_SWAP_ALGORITHM'],
                        envs['MXNET_PREFETCH_ALGORITHM'],
                        envs['MXNET_PREFETCH_STEP_AHEAD'])
        options = ['--num_gpus={}'.format(args.num_gpus),
                   '--num_loops={}'.format(args.num_loops),
                   '--batch_size={}'.format(args.batch_size)]

    else:
        raise NotImplementedError
    options.append(args.model)
    proc = subprocess.Popen(['python', 'benchmark.py'] + options, env=envs,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1, universal_newlines=True,
                            preexec_fn=os.setsid)
    register_sigint(proc)
    with open(log_name, 'w') as fp:
        while True:
            line = proc.stdout.readline()
            if line:
                if line[:8] == "duration":
                    tmp = re.findall('\d+\.?\d*', line)
                    # Duration, Average, Std
                    result = [True, tmp[0], tmp[1], tmp[2]]
                if args.also_print:
                    print(line, end='')
                fp.write(line)
            else:
                break
        stdout, stderr = proc.communicate()
        message = 'The program exits with returncode = {}\n'\
                    .format(proc.returncode)
        if proc.returncode != 0:
            name = 'None'
            for k, v in signal.__dict__.items():
                if k.startswith('SIG') and not k.startswith('SIG_'):
                    if getattr(signal, k) == -proc.returncode:
                        name = k
                        break
            if name == 'SIGSEGV':
                message += 'Segmentation Fault.\n'
            else:
                message += name + ' : ' + str(proc.returncode) + '\n'
            result = [False, proc.returncode, name]
        else:
            fp.write(message)
        print(message, end="")
    return result

# Self Test: Ignore layer, wide scale, batch size, algorithm settings
def run_self_test(args):
    envs = {val[0]: val[1] for val in (s.split('=') for s in args.envs)}
    #layers = [50, 101, 152, 200, 269]
    #batches = [1, 4, 8, 16, 32, 64, 128]
    #widths = range(1,6)
    #engines = ['NaiveEngine', 'ThreadedEngine']
    layers = [269]
    batches = [1,8]
    widths = [1,6]
    engines = ['ThreadedEngine']
    lst = [(layer, batch, width, engine)
            for layer in layers 
            for batch in batches
            for width in widths
            for engine in engines
          ]
    with open('log_selftest.csv', 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['layer','batch_size','wide_scale','engine',
                         'duration','average','std',
                         'returncode','error_name'])
        for (layer, batch, width, engine) in lst:
            print("Testing argument:",layer, batch, width, engine)
            args.num_layers = layer
            args.batch_size = batch
            args.wide_scale = width
            envs['MXNET_ENGINE_TYPE']=engine
            result = run_script(args,envs)
            print("result:", result)
            if result[0] == True:
                writer.writerow([layer, batch, width, engine]
                                + result[1:] + [0])
            else:
                writer.writerow([layer, batch, width, engine]
                                + ['','',''] + result[1:])


class DumpArguments(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        nargs=0
        super(DumpArguments, self).__init__(option_strings, dest, nargs=nargs,
                                            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        path = os.path.join('.', 'config_' + namespace.model)
        if os.path.exists(path):
            print('{} will be overwitten. '
                  'Do you want to continue? (y/N) '.format(path))
            answer = sys.stdin.readline().strip().lower()
            if answer != 'y':
                sys.exit(0)
        with open(path, 'w') as fp:
            fp.write(getattr(namespace, 'model') + '\n')
            fp.write('--envs\n')
            fp.write('\n'.join(getattr(namespace, 'envs')) + '\n')
            for arg in sorted(vars(namespace)):
                if arg != 'model' and arg != 'dump' and arg != 'envs':
                    val = getattr(namespace, arg)
                    if type(val) == bool:
                        fp.write('--' + ('' if val else 'no') +  arg + '\n')
                    else:
                        fp.write('--' + arg + '\n')
                        fp.write(str(val) + '\n')
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser('Benchmark Tests',
                                     fromfile_prefix_chars='@')
    parser.add_argument('model', type=str, default='resnet',
                        help='The model to be tested.')
    parser.add_argument('--dump', type=bool, action=DumpArguments)
    parser.add_argument('--also_print', action='store_true')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs.')
    parser.add_argument('--num_loops', type=int, default=10,
                        help='Number of loops.')
    # Model settings
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size.')
    parser.add_argument('--num_layers', type=int, default=152,
                        help='Number of layers (for some models only).')
    parser.add_argument('--wide_scale', type=int, default=1,
                        help='Wide scale (for W-ResNet only).')
    parser.add_argument('--self_test', action='store_true',
                        help='Self test, ignoring some of the other arguments')
    # Environment variables settings.
    # NOTE that if b new environment variable is added, it should also be here.
    parser.add_argument('--envs', type=str, nargs='+',
                        help='Environment variables.',
                        default=['PYTHONPATH=/home/fegin',
                                 'LD_LIBRARY_PATH=/usr/local/cuda/lib64',
                                 'CUDA_VISIBLE_DEVICES=0',
                                 'MXNET_ENGINE_TYPE=NaiveEngine',
                                 'MXNET_PREFETCH_ALGORITHM=ComputePrefetch',
                                 'MXNET_PREFETCH_STEP_AHEAD=30',
                                 'MXNET_MEM_MGR_TYPE=CUDA',
                                 'MXNET_SWAP_ALGORITHM=NaiveHistory',
                                 'MXNET_SWAP_ASYNC=1',
                                 'MXNET_ADAPTIVE_HISTORY=0',
                       ])
    args, _ = parser.parse_known_args()
    print('Arguments: ')
    for arg in sorted(vars(args)):
        if arg != 'envs':
            print('    {}: {}'.format(arg, str(getattr(args, arg))))
        else:
            print('    envs:')
            pprint.pprint(getattr(args, arg), indent=8)
    if args.self_test:
        print('Starting selftest for model {}...'.format(args.model))
        run_self_test(args)
    else:
        envs = {val[0]: val[1] for val in (s.split('=') for s in args.envs)}
        run_script(args, envs)

if __name__ == '__main__':
    main()
