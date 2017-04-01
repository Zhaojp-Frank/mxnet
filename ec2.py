import argparse
import os
import subprocess
from threading import Thread


def DoSSH(ip, command, log_path):
    def run(ip, prog):
        path_out = os.path.join(log_path, ip + '_stdout.txt')
        path_err = os.path.join(log_path, ip + '_stderr.txt')
        print(ip)
        with open(path_out, 'w') as out_fp:
            with open(path_err, 'w') as err_fp:
                subprocess.check_call(prog, stdout=out_fp, stderr=err_fp,
                                      shell=True)

    prog = 'ssh -o StrictHostKeyChecking=no ' + ip + ' \'' + command + '\''
    thread = Thread(target=run, args=(ip, prog,))
    thread.setDaemon(True)
    thread.start()
    return thread


def DoRun(script, master, all_ips, log_path):
    threads = []
    for idx, ip in enumerate(all_ips):
        threads.append(DoSSH(ip, script, log_path))
    for t in threads:
        t.join()
    return threads


def DoIndexedRun(script, master, all_ips, log_path):
    threads = []
    for idx, ip in enumerate(all_ips):
        threads.append(DoSSH(ip, script % idx, log_path))
    for t in threads:
        t.join()
    return threads


def DoMount(master, all_ips, log_path):
    threads = []
    command = "sudo mount -rw %s:/home/ubuntu /home/ubuntu" % master
    for ip in all_ips:
        if ip != master:
            threads.append(DoSSH(ip, command, log_path))
    for t in threads:
        t.join()
    return threads


def GetAllIPs(master):
    # Get all public_ip and private_ip via ec2 tool.
    command = \
        ('aws ec2 describe-instances --query ' +
         '"Reservations[*].Instances[*].[PublicIpAddress, PrivateIpAddress]" ' +
         '--output=text')
    ips = os.system(command)

    all_ips = [master]
    with open('all_ips', 'w') as fp:
        fp.write(master + '\n')
        with os.popen(command) as ips:
            for line in ips:
                ips = line.strip().split()
                public_ip = ips[0]
                private_ip = ips[1]
                if private_ip != "None" and private_ip != master :
                    fp.write(private_ip + '\n')
                    all_ips.append(private_ip)
    print(all_ips)
    return all_ips


def main():
    parser = argparse.ArgumentParser(description='Run commands.')
    parser.add_argument('-m', '--master', type=str,
                        help='The address of master')
    parser.add_argument('-l', '--log_path', type=str, default='./',
                        help='Log path')
    parser.add_argument('--mount', action='store_true',
                        help='Whether to do mount or not.')
    parser.add_argument('--run', type=str, help='Run command.')
    parser.add_argument('--run_indexed', type=str,
                        help='Run command (must contains %%d) with index ' +
                             'which will be assigned by this Python program.')
    parser.add_argument('--test', action='store_true',
                        help='Fake ips without ec2.')
    args = parser.parse_args()

    if args.test:
        master = '127.0.0.1'
        all_ips = ['127.0.0.1', 'localhost']
    else:
        master = args.master
        assert args.master
        all_ips = GetAllIPs(args.master)
    if args.mount:
        DoMount(args.master, all_ips, args.log_path)
    if args.run:
        DoRun(args.run, args.master, all_ips, args.log_path)
    if args.run_indexed:
        DoIndexedRun(args.run_indexed, args.master, all_ips, args.log_path)


if __name__ == "__main__":
    main()

