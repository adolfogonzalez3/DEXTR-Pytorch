##
If you only wish to train the models then continue otherwise you must pull the
repository from github in order to obtain the pretrained weights for densenet,
shufflenet, and squeezenet.
##
In order to get the demos running you must first have conda installed.

Once you have installed conda you must install the conda pytorch package
conda install pytorch torchvision

Then install the package using pip
pip install -e .

Now you can run the training for the models by running.
python train_densenet.py
or
python train_shufflenet.py
or
python train_squeezenet.py

##
If you have the pretrained weights or have trained your own models then you can
run them by using
python demo_densnet.py
or
python demo_shufflenet.py
or
python demo_squeezenet.py
