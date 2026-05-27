import subprocess
# This file is a step by step guide of training, exporting and evaluating a model
# %% First train a model
print("**********************************************************************************************************************************")
print("Starting training...")
print("**********************************************************************************************************************************")
subprocess.run("python train_sc.py", shell=True,executable="/bin/bash", stderr=subprocess.STDOUT)
# This will train a model and save it in the results_journal folder
# %% Now export the model
print("**********************************************************************************************************************************")
print("Starting exporting...")
print("**********************************************************************************************************************************")
subprocess.run("python model_export.py", shell=True, executable="/bin/bash", stderr=subprocess.STDOUT)
# This will export the model to the export_model folder
# %% Now convert the model to C++ code
print("**********************************************************************************************************************************")
print("Starting conversion...")
print("**********************************************************************************************************************************")
subprocess.run("python model_cpp_convert.py", shell=True, executable="/bin/bash", stderr=subprocess.STDOUT)
# This will convert the model to C++ code and save it in the cpp_code folder, and then compile it
# %% Now evaluate the cpp model
print("**********************************************************************************************************************************")
print("Starting evaluation...")
print("**********************************************************************************************************************************")
subprocess.run("python test_sc_cpp_impl.py", shell=True, executable="/bin/bash", stderr=subprocess.STDOUT)
# This will evaluate the cpp model and save the results in the results folder
# Finally, the confusion matrix will be saved in the root folder as a png file