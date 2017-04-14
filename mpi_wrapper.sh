#/!bin/bash

if [ `hostname` == $MASTER ]; then
  $@ 2>&1 | tee $LOG_DIR/`hostname`.log
else
  $@ &> $LOG_DIR/`hostname`.log
fi
