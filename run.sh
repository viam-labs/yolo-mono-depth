#!/bin/bash
cd `dirname $0`

# On Linux aarch64 (Pi / ARM SBCs), install CPU torch first so
# `ultralytics` does not pull CUDA wheels (nvidia-cudnn, etc.).
_install_reqs() {
  if [ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "aarch64" ]; then
    pip3 install --upgrade torch torchvision \
      --index-url https://download.pytorch.org/whl/cpu
  fi
  pip3 install --upgrade -r requirements.txt
}

if [ -f .installed ]
  then
    source viam-env/bin/activate
  else
    python3 -m pip install --user virtualenv --break-system-packages
    python3 -m venv viam-env
    source viam-env/bin/activate
    _install_reqs
    if [ $? -eq 0 ]
      then
        touch .installed
    fi
fi

# Be sure to use `exec` so that termination signals reach the python process,
# or handle forwarding termination signals manually
exec python3 -m src $@
