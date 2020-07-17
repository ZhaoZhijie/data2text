import paramiko
from scp import SCPClient
import os

HOST = "35.189.123.190"
PORT = 22
USER = "root"
PASS = "17951"

def scp_files(src_path, tar_path):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh_client.connect(HOST, PORT, USER, PASS)
    scpclient = SCPClient(ssh_client.get_transport(),socket_timeout=15.0)
    if not isinstance(src_path, list):
        src_path = [src_path]
    for file in src_path:
        try:
            scpclient.put(file, tar_path)
        except Exception as e:
            print(e)
            print("File {} upload error".format(file))
        else:
            print("file {} upload successfully".format(file))
    ssh_client.close()

def scp_models(seed, exp):
    path = "experiments/exp-seed-{}/exp-{}/models".format(seed, exp)
    files = os.listdir(path)
    paths = [os.path.join(path, f) for f in files]
    tar_path = "home/zzjstars/zzj{}n{}_drive/".format(exp.lower(), seed)
    scp_files(paths, tar_path)


seed = sys.argv[1]
exp = sys.argv[2]
scp_models(seed, exp)

# scp_files(["/mnt/c/Users/Administrator/Desktop/summary.pdf"], "/home/zzjstars")





