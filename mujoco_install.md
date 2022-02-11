# Mujoco installation guide

## Download Mujoco and install

```$ wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz```

```$ mkdir ~/.mujoco```

```$ mv mujoco210-linux-x86_64.tar.gz ~/.mujoco```

### Unpack in mujoco folder

```$ cd ~/.mujoco && tar -xf mujoco210-linux-x86_64.tar.gz```

### Modify bashrc
Add the following lines to your .bashrc. Replace `<user>` with your user name.

```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<user>/.mujoco/mujoco210/bin```

```export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia```

Source after you are done!

```$ source ~/.bashrc```

### Install python bindings

```$ pip install mujoco-py```

As an alternative, install all packages from the conda environment yaml:

```$ conda env create -f environment.yml```

## Patchelf
If mujoco_test.py is exiting with the following error

```FileNotFoundError: [Errno 2] No such file or directory: 'patchelf'```

you have to install patchelf. 

```$ cd ~/ && git clone https://github.com/NixOS/patchelf.git```

```$ cd patchelf && ./bootstrap.sh```

```$ ./configure --prefix=$HOME/.local```

```$ make```

```$ make install```

```$ rm -rf ~/patchelf```

Add .local/bin to PATH in .bashrc. Make sure to replace `<user>` with your user name.

```export PATH=$PATH:/home/<user>/.local/bin```

Don't forget to source.

```$ source ~/.bashrc```

## Done

You should now be able to run mujoco_test.py
