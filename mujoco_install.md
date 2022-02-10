# Mujoco installation guide

## Download mujoco and install

wget https://mujoco.org/download/mujoco210-linux-x86\_64.tar.gz
mkdir ~/.mujoco
mv mujoco210-linux-x86\_64.tar.gz ~/.mujoco

Move to .mujoco

tar -xf mujoco210-linux-x86\_64.tar.gz

### Modify bashrc
Add the following lines to your .bashrc. Don't forget to source once you are done.
export LD\_LIBRARY\_PATH=$LD\_LIBRARY\_PATH:/home/mschuck/.mujoco/mujoco210/bin
export LD\_LIBRARY\_PATH=$LD\_LIBRARY\_PATH:/usr/lib/nvidia

### Install python bindings
pip install mujoco-py

## Possible bugs
Try to run mujoco\_test.py The following errors can be resolved by installing additional tools/libraries
FileNotFoundError: [Errno 2] No such file or directory: 'patchelf'
-> Install patchelf with apt-get install patchelf or from source 
