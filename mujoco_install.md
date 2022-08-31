# Mujoco installation guide

## Download Mujoco and install

```$ wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz```

```$ mkdir ~/.mujoco```

```$ mv mujoco210-linux-x86_64.tar.gz ~/.mujoco```

### Unpack in mujoco folder

```$ cd ~/.mujoco && tar -xf mujoco210-linux-x86_64.tar.gz```

### Modify bashrc
Execute the following commands to add environment variables required by Mujoco to your .bashrc. Replace `<user>` with your user name.

```$ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<user>/.mujoco/mujoco210/bin' >> ~/.bashrc```

```$ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc```

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

```$ echo 'export PATH=$PATH:/home/<user>/.local/bin' >> ~/.bashrc```

Don't forget to source.

```$ source ~/.bashrc```

## GLEW missing GL version
Sometimes rendering the simulation throws the error

```GLEW initalization error: Missing GL version```

This can be resolved by adding an additional environment variable to your .bashrc (source after you are done!).

```$ echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so' >> ~/.bashrc```

>**Note**: Depending on the installed version, it may also be required to **remove** this line. If you get an error with this line in your bashrc, try removing it.

## GL/osmesa.h: No such file or directory
If this error occures, you are missing libraries for the cython extension compilation. Install the necessary libraries with

```$ sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3```

## Done

You should now be able to run mujoco_test.py
