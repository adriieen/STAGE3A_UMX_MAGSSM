import os

try:
    print('Virtual enviourment at:', os.environ['VIRTUAL_ENV'])

except KeyError:
    print('No virtual environment detected. Please activate your virtual environment.')
    exit(1)


import sys
import os
import subprocess
import sysconfig

# check if python-devel is installed
if not os.path.exists(os.path.join(sysconfig.get_config_var('INCLUDEPY'), 'Python.h')):
    print("ERROR: Python development headers not found. Please install python3-devel or python3-dev package.",file=sys.stderr)
    print("ERROR: On Ubuntu/Debian: sudo apt install python3-dev",file=sys.stderr)
    print("ERROR: On Fedora/RHEL: sudo dnf install python3-devel",file=sys.stderr)
    print("ERROR: Required for building the C++ part of the project. Mannually disable this check if you are not building the C++ part.",file=sys.stderr)
    sys.exit()
else:
    print("Python development headers found. Proceeding with the installation.")



# Upgrade pip
subprocess.check_call(['git', 'submodule', 'init'])
subprocess.check_call(['git', 'submodule', 'update'])

# Upgrade pip
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])

# Install required packages
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'req.txt'])

import shutil

import cosine_annealing_warmup

shutil.copy2('./patches/scheduler.py', cosine_annealing_warmup.__path__[0])