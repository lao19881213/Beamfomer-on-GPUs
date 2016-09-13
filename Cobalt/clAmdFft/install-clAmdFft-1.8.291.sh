#!/bin/bash
#
# Installation script

# Grab the name of the bash script, which must have a certain format: install-${AppName}-${AppVersion}.sh
ScriptName=(${0//-/ })
AppName="${ScriptName[1]}"

# Parse this version number from the name of this script
AppVersionArray=(${ScriptName[2]//./ })
AppVersion="${AppVersionArray[0]}.${AppVersionArray[1]}.${AppVersionArray[2]}"

# Check for any switch arguments
doneswitches=0
badswitch=0
acceptlicense=0
installdir=""
while test "${doneswitches}" = "0"
do
  case "${1-}" in
    -accept*)
       acceptlicense=1
       shift;;
    -installdir=*)
       installdir=`echo ${1} | sed -e 's%.*=%%'`
       shift;;
    -*)
       echo "${ScriptName}: unrecognised switch: ${1}"
       badswitch=1
       exit
       shift;;
     *)
       doneswitches=1;;
  esac
done

showLicense()
{
if [ "${acceptlicense}" = "1" ]; then
  echo "Warning: by installing this software you have accepted"
  echo "the license agreement in ${AppName}-EULA.txt"
  reply="accept"
else
  more ${AppName}-EULA.txt

  reply=""
  while [ "${reply}" != "accept" -a "${reply}" != "decline" ]; do
    echo -e "[accept/decline]? : \c"
    read reply
    reply=`echo ${reply} | tr [:upper:] [:lower:]`
  done
fi
}

get_yes_no()
{
reply=""
while [ "$reply" != "y" -a "$reply" != "n" ]; do
  echo -e "$1 ? [y/n] : \c"
  read reply
  reply=`echo ${reply} | tr [:upper:] [:lower:]`
done
}

echo -e "                   ${AppName}-${AppVersion} Installation  "
echo -e "                   =====================================  "
echo -e ""
echo -e "This script will install ${AppName} version ${AppVersion}"

showLicense

if [ "${reply}" != "accept" ]; then
  echo "Installation declined. ${AppName}-${AppVersion} not installed."
  exit
fi

echo -e ""
echo -e "Where do you want to install ${AppName}-${AppVersion}?  Press return to use"
echo -e "the default location (/opt/${AppName}-${AppVersion}), or enter an alternative path."
echo -e "The directory will be created if it does not already exist."
if [ "${installdir}" != "" ]; then
  INSTALLDIR=${installdir}
else
  INSTALLDIR=""
  while [ "${INSTALLDIR}" = "" ]; do
    echo -e "> \c"
    read ans
    if [ $ans ]
    then
        case $ans in
        *) INSTALLDIR=$ans ;;
        esac
    else
        INSTALLDIR=/opt/${AppName}-${AppVersion}
    fi
  done
fi

# Replace any ~ by ${HOME} otherwise you end up with a
# subdirectory named ~ (dangerous if you then try to remove it!)
INSTALLDIR=`echo ${INSTALLDIR} | sed -e "s%~%${HOME}%g"`

echo -e ""
echo -e "Installing to : ${INSTALLDIR}"
echo -e ""

if [ ! -d "${INSTALLDIR}" ]
then
  mkdir -p "${INSTALLDIR}"
  if [ $? -ne 0 ]
  then
    echo -e "***** Cannot create installation directory, installation failed *****"
    exit
  fi
fi

# Extract everything from the compressed tar file
fromdir=$( pwd )
cd "${INSTALLDIR}"
tar -xvf "${fromdir}/${AppName}-${AppVersion}-Linux.tar.gz"

echo -e ""
echo -e "====== ${AppName}-${AppVersion} installation complete ======"
