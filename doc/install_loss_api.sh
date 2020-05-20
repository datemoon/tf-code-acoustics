# Install mmi,smbr,mpfe tensorflow api.
# You must be set up TENSORFLOW_SRC_PATH and the dir must be have libtensorflow_framework.so
# export TENSORFLOW_SRC_PATH=/usr/local/python35/lib/python3.5/site-packages/tensorflow

cd ../fst/cc
make;
python3 setup.py install 

cd -
# Install chain loss tensorflow api.
cd ../kaldi_2_tf_io/
make -j 3;make static
python3 setup.py install

# If there is not error, you succeed.
