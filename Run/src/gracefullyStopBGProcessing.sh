#!/bin/bash

source locations.sh

set_psetinfo

echo "cancel all" > /dev/tcp/$FIRSTPSET/4000 2>/dev/null &&
echo "quit"       > /dev/tcp/$FIRSTPSET/4000 2>/dev/null &&
sleep 10 # allow processes to quit
