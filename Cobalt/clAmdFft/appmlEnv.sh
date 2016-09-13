#! /bin/bash
# Short script meant to automate the task of setting up a terminal window to 
# use the APPML library

# Verify that this script has been sourced, not directly executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]
then
	echo "This script is meant to be sourced '.', as it modifies environmental variables"
	echo "Try running as: '. $(basename ${0})'"
	exit
fi

# This is a sequence of bash commands to get the directory of this script
scriptDir=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
# echo Script dir is: ${scriptDir}

# Bash regexp to determine if the terminal is set up to point to APPML
if [[ ${LD_LIBRARY_PATH} = *${scriptDir}/lib64:${scriptDir}/lib32* ]]
then
	echo "APPML math libraries is set in LD_LIBRARY_PATH"
else
	echo "Patching LD_LIBRARY_PATH to include APPML math libraries"
	export LD_LIBRARY_PATH=${scriptDir}/lib64:${scriptDir}/lib32:${LD_LIBRARY_PATH}
fi
