#!/bin/bash
CUR_DIR=$(cd $(dirname $0); pwd)

cd $CUR_DIR

python3 "$CUR_DIR/cnn.py" "$@"
