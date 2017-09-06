import argparse
import os
import random
import subprocess
import time
from threading import Thread


def DoSSH(ip, command, log_path, index=None):
    def run(ip, prog):
        suffix = '_' + str(index) if index is not None else ''
        path_out = os.path.join(log_path, ip + suffix + '_stdout.txt')
        path_err = os.path.join(log_path, ip + suffix + '_stderr.txt')
        print(ip)
        with open(path_out, 'w') as out_fp:
            with open(path_err, 'w') as err_fp:
                try:
                    subprocess.check_call(prog, stdout=out_fp, stderr=err_fp,
                                          shell=True)
                except Exception as e:
                    print(suffix)
                    print(e)

    prog = 'ssh -t -o StrictHostKeyChecking=no ' + ip + ' \'' + command + '\''
    thread = Thread(target=run, args=(ip, prog,))
    thread.setDaemon(True)
    thread.start()
    return thread


def DoRun(script, master, all_ips, log_path):
    threads = []
    same_ip = all(x == all_ips[0] for x in all_ips)
    for idx, ip in enumerate(all_ips):
        if same_ip:
            threads.append(DoSSH(ip, script, log_path, idx))
            time.sleep(random.randint(1, 20) / 10.0)
        else:
            threads.append(DoSSH(ip, script, log_path))
    for t in threads:
        t.join()
    return threads


def DoIndexedRun(script, master, all_ips, log_path):
    threads = []
    same_ip = all(x == all_ips[0] for x in all_ips)
    for idx, ip in enumerate(all_ips):
        if same_ip:
            threads.append(DoSSH(ip, script % idx, log_path, idx))
            time.sleep(random.randint(1, 20) / 10.0)
        else:
            threads.append(DoSSH(ip, script % idx, log_path))
    for t in threads:
        t.join()
    return threads


def DoMount(master, all_ips, mount_path, log_path):
    threads = []
    command = "cd /home && sudo mount -rw %s:%s %s" % (master, mount_path, mount_path)
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
    parser.add_argument('--mount_path', type=str,
                        help='The path to do nfs mount.')
    parser.add_argument('--run', type=str, help='Run command.')
    parser.add_argument('--run_indexed', type=str,
                        help='Run command (must contains %%d) with index ' +
                             'which will be assigned by this Python program.')
    parser.add_argument('--host', type=str,
                        help='Hosts.')
    args = parser.parse_args()

    if args.host:
        all_ips = []
        print (args.host)
        master = '127.0.0.1'
        with open(args.host) as fp:
            for line in fp:
                loc = line.find(':')
                if loc != -1:
                    all_ips.append(line[0:loc])
                else:
                    all_ips.append(line.strip())
        print(all_ips)
    else:
        master = args.master
        assert args.master
        all_ips = GetAllIPs(args.master)
    if args.mount:
        DoMount(args.master, all_ips, args.mount_path, args.log_path)
    if args.run:
        DoRun(args.run, args.master, all_ips, args.log_path)
    if args.run_indexed:
        DoIndexedRun(args.run_indexed, args.master, all_ips, args.log_path)


if __name__ == "__main__":
    main()

