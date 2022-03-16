FROM python:3.9

# install MPI
RUN apt-get update
RUN apt-get install -y libopenmpi-dev openmpi-bin openmpi-doc

# RUN pip install --upgrade pip
# Avoid cache miss after change in ADD directory and no changes to reqirements.txt
COPY requirements.txt /var
RUN pip install -r /var/requirements.txt
# Mujoco_py throws permission denied errors if executed as non-root user becuase of a lockfile without access rights
RUN pip install patchelf
# Install mujoco
WORKDIR /usr/.mujoco
RUN wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
RUN tar -xf mujoco210-linux-x86_64.tar.gz
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/.mujoco/mujoco210/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/.mujoco/mujoco210/bin
ENV MUJOCO_PY_MUJOCO_PATH /usr/.mujoco/mujoco210
RUN apt-get update
RUN apt-get install libosmesa6-dev -y
# Trigger mujoco_py cython compilation so it is cached on image build instead of image run
RUN python -c "import mujoco_py"
RUN chmod -R 777 /usr/local/lib/python3.9/site-packages/mujoco_py
# Prevent OPENMPI error within Docker. See https://github.com/open-mpi/ompi/issues/4948
ENV OMPI_MCA_btl_vader_single_copy_mechanism none

# Go into workdir, install package
ADD . /mnt
WORKDIR /mnt
RUN pip install .

# Switch to user 1001 because MPI execution with root user is discouraged
USER 1001
ENTRYPOINT [ "tail" ]
# ENTRYPOINT ["python", "./test.py"]