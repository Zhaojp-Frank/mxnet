#/!bin/bash

if [ `hostname` == $MASTER ]; then
  $@  | tee $LOG_DIR/`hostname`.log
else
  $@ > $LOG_DIR/`hostname`.log
fi
