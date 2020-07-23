import paramiko
from scp import SCPClient
import os
import sys
from logger import logger

HOST = "35.189.123.190"
PORT = 22
USER = "root"
PASS = "17951"

def scp_files(src_path, tar_path, get=False):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh_client.connect(HOST, PORT, USER, PASS)
    scpclient = SCPClient(ssh_client.get_transport(),socket_timeout=15.0)
    succs = []
    fails = []
    if not isinstance(src_path, list):
        src_path = [src_path]
    for file in src_path:
        try:
            if get:
                scpclient.get(file, tar_path)
            else:
                scpclient.put(file, tar_path)
            
        except Exception as e:
            logger.info(e)
            logger.info("File {} copy error".format(file))
            fails.append(file)
        else:
            logger.info("file {} copy successfully".format(file))
            succs.append(file)
    ssh_client.close()
    return succs, fails


def scp_models(seed, exp):
    path = "experiments/exp-seed-{}/exp-{}/models".format(seed, exp)
    files = os.listdir(path)
    paths = [os.path.join(path, f) for f in files]
    tar_path = "/home/zzjstars/zj17501_drive/zjmodels/exp-seed-{}/exp-{}/models/".format(seed, exp)
    scp_files(paths, tar_path)


if __name__ == "__main__":
    src = sys.argv[1]
    tar = sys.argv[2]
    get = sys.argv[3] == "True"
    scp_files(src, tar, get)

# scp_files(["/mnt/c/Users/Administrator/Desktop/summary.pdf"], "/home/zzjstars")





