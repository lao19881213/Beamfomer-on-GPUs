#
# This script should be used to create and deploy a new release branch for OLAP. The script
# can also be used to rebuild en redeploy an existing release.
#
# A typical evocation:
#   lofarsys@bgfen0:~> RELEASE=current make
#
# All variables below which are set using ?= are meant to be
# optionally overruled.
#
# Various useful commands. RELEASE must be one of:
#   - set to a date denoting a previous release
#   - "new"     -- create a new release, stamped today
#   - "current" -- denotes the release currently active on bgfen.
#   - "trunk"   -- no release will be created, but the trunk
#                  will be rolled out directly.
#
#   make install                - compiles and installs the sources (default)
#   make activate               - activate the install
#   make all                    - make install + make activate
#
#   make restart                - restart OLAP (swlevel 0 && swlevel 6)
#
#   make prev                   - switch back to the previous release
#   make sync                   - sync the locus nodes to the current release
#
# Other optional environment settings:
#   LOCUS_NODES                 - the set of storage nodes to act on (cexec notation, i.e. :0-1 or locus: (default))
#
#   COMMON_FLAGS                - extra CMake flags to use for all packages
#   ION_FLAGS                   - extra CMake flags to use for IONProc compilation
#   CN_FLAGS                    - extra CMake flags to use for CNProc compilation
#   BGFEN_FLAGS                 - extra CMake flags to use for building software running on bgfen0/1 (ApplController, ACDaemon, RTCP/Run, etc)
#   STORAGE_FLAGS               - extra CMake flags to use for Storage compilation
#
#   BUGNR                       - the bugzilla bug number under which to file the new release branch checkin
#   DEVELOPER                   - the subversion user which commits a new relase branch
#

ifeq "${RELEASE}" "new"
  RELEASE := $(shell date +%Y%m%d)
endif

ifeq "${RELEASE}" "current"
  RELEASE := $(shell basename `readlink -f ${HOME}/production/lofar` 2>/dev/null | cut -d- -f2)
endif

ifeq "${RELEASE}" ""
  $(error $$RELEASE was not set, or could not determine current install. Use export RELEASE=`date +%Y%m%d` for a new release or export RELEASE=current to work with the current release)
endif

ifneq "${USER}" "lofarsys"
  # the software needs to be installed as user lofarsys
  $(error Please run as lofarsys)
endif

ifneq "${HOST}" "bgfen0"
  ifneq "${HOST}" "bgfen1"
    # the software needs to be installed on bgfen0 and/or bgfen1
    $(error Please run on bgfen0 or bgfen1)
  endif
endif

ifeq "$(shell basename ${HOME})" "lofartest"
  TESTENV := 1
endif

# The developer under which a release is branched off
DEVELOPER ?= mol

# The bug number under which the release branch is filed
BUGNR ?= 1362

# Compilation flagsq
COMMON_FLAGS += -DLOFAR_SVN_UPDATE=ON -DBUILD_TESTING=OFF
ION_FLAGS +=
CN_FLAGS +=
BGFEN_FLAGS += -DUSE_LOG4CPLUS=ON
STORAGE_FLAGS += -DDAL_ROOT_DIR=/globalhome/lofarsystem/packages/root/lhn/lda -DUSE_MPI=OFF -DUSE_LOG4CPLUS=ON

# Extra powers for Storage_main
STORAGE_CAPABILITIES = cap_sys_admin,cap_sys_nice,cap_ipc_lock

# Subversion configuration
SVN = svn --non-interactive --username ${DEVELOPER}
SVN_TRUNK = https://svn.astron.nl/LOFAR/trunk
ifeq "${RELEASE}" "trunk"
  SVN_BRANCH = ${SVN_TRUNK}
else  
  SVN_BRANCH ?= https://svn.astron.nl/LOFAR/branches/LOFAR-Release-${RELEASE}
endif  

# Directory structure
SOURCE_DIR = ${HOME}/source/lofar-${RELEASE}
BUILD_DIR = ${HOME}/build/lofar-${RELEASE}
INSTALL_DIR ?= ${HOME}/production/lofar-${RELEASE}

ifdef TESTENV
  LOCUS_HOME = /data/home/lofartest
else
  LOCUS_HOME = /data/home/lofarsys
endif

LOCUS_INSTALL_DIR ?= ${LOCUS_HOME}/production/lofar-${RELEASE}

# Host names
LHN_HOST = lhn001.cep2.lofar
ifeq "${HOST}" "bgfen0"
  LOCUS_NODES ?= locus:0-93
else
  LOCUS_NODES ?= locus:94-99
endif

# Determine if the requested release is already active
CURRENT_INSTALL := $(shell readlink -f ${HOME}/production/lofar)
PREV_INSTALL    := $(shell readlink -f ${HOME}/production/lofar-prev)
MY_INSTALL      := ${INSTALL_DIR}

# Descriptive names for targets
CHECKOUT      = ${SOURCE_DIR}/LOFAR/CMakeLists.txt
INSTALL_ION   = ${INSTALL_DIR}/bgpion/bin/ION_Processing
INSTALL_CN    = ${INSTALL_DIR}/bgpcn/bin/CN_Processing
INSTALL_LHN   = ${INSTALL_DIR}/lhn/bin/Storage_main
INSTALL_BGFEN = ${INSTALL_DIR}/bgfen/bin/ApplController

.DEFAULT:	install
.PHONY:		all install activate prev sync
all:		install activate
install:	install_ion install_cn install_bgfen install_lhn
activate:	activate_bgfen activate_locus
prev:		prev_bgfen prev_locus
sync:		activate_locus

.PHONY:		checkout configure install_ion install_cn install_bgfen install_lhn

checkout:	${CHECKOUT}
install_ion:	${INSTALL_ION}
install_cn:	${INSTALL_CN}
install_bgfen:	${INSTALL_BGFEN}
install_lhn:	${INSTALL_LHN}

.PHONY:		restart
restart:
	swlevel 0 && swlevel 6

.PHONY:		clean
clean:
	rm -rf ${BUILD_DIR} ${SOURCE_DIR} # never clean install


# Create a release branch, if it does not yet exist.
.PHONY:		release_branch
release_branch:
	${SVN} info ${SVN_BRANCH} &>/dev/null || ${SVN} copy -m "bug 1362: New release: ${RELEASE}" ${SVN_TRUNK} ${SVN_BRANCH}

# Checkout the release branch into the source directory, but only LOFAR/* and LOFAR/CMake/*, as cmake will pull in the rest.
${CHECKOUT}:		release_branch
	mkdir -p ${SOURCE_DIR}
	test -d ${SOURCE_DIR}/LOFAR || ${SVN} checkout -q --depth=files ${SVN_BRANCH} ${SOURCE_DIR}/LOFAR
	${SVN} update -q --depth=files ${SOURCE_DIR}/LOFAR
	${SVN} update -q ${SOURCE_DIR}/LOFAR/CMake

# Configure all packages
# cmake needs to pull in additional packages from subversion. A parallel build would pull the same directories to the same destinations in parallel,
# which subversion doesn't allow.
configure:		checkout
	mkdir -p ${BUILD_DIR}/bgpion_opt
	cd ${BUILD_DIR}/bgpion_opt && cmake ${SOURCE_DIR}/LOFAR -DBUILD_PACKAGES=IONProc ${COMMON_FLAGS} ${ION_FLAGS} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/bgp_ion

	mkdir -p ${BUILD_DIR}/bgpcn_opt
	cd ${BUILD_DIR}/bgpcn_opt && cmake ${SOURCE_DIR}/LOFAR -DBUILD_PACKAGES=CNProc ${COMMON_FLAGS} ${CN_FLAGS} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/bgp_cn

	mkdir -p ${BUILD_DIR}/bgfen/gnu_opt
	cd ${BUILD_DIR}/bgfen/gnu_opt && cmake ${SOURCE_DIR}/LOFAR -DBUILD_PACKAGES="Appl_Controller;APLCommon;Run" ${COMMON_FLAGS} ${BGFEN_FLAGS} -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/bgfen

	mkdir -p ${BUILD_DIR}/lhn/gnu_opt
        # remove /data/sys: on lhn, /opt is a symlink for /data/sys/opt, which cmake resolves for us. However, on locus,
        # we really need it as /opt. 
	ssh ${LHN_HOST} 'cd ${BUILD_DIR}/lhn/gnu_opt && cmake ${SOURCE_DIR}/LOFAR -DBUILD_PACKAGES="Storage;StaticMetaData" ${COMMON_FLAGS} ${STORAGE_FLAGS} -DCMAKE_INSTALL_PREFIX=${LOCUS_INSTALL_DIR}'

# Build and install ION_Processing
${INSTALL_ION}:		configure
	cd ${BUILD_DIR}/bgpion_opt && make -j16 install

# Build and install CN_Processing
${INSTALL_CN}:		configure
	cd ${BUILD_DIR}/bgpcn_opt && make -j16 install

# Build and install support software on bgfen
${INSTALL_BGFEN}:	configure
	cd ${BUILD_DIR}/bgfen/gnu_opt && make -j16 install
	- cd ${HOME}/production && cp ${PREV_INSTALL}/bgfen/etc/*.debug ${INSTALL_DIR}/bgfen/etc
	- ln -sfT ${HOME}/log ${INSTALL_DIR}/bgfen/log

# Build and install Storage_main
${INSTALL_LHN}:		configure
	ssh ${LHN_HOST} 'cd ${BUILD_DIR}/lhn/gnu_opt && make -j16 install'
	ssh ${LHN_HOST} 'for d in `ldd ${LOCUS_INSTALL_DIR}/bin/Storage_main | awk "/=> \// { print \\$$3; }"`; do echo "Scanning dependency $$d"; if [ "`stat -f -c %T $$d`" == "nfs" ]; then echo "Copying dependency $$d from NFS to local disk"; cp $$d ${LOCUS_INSTALL_DIR}/lib; fi; done'
	ssh ${LHN_HOST} "cexec ${LOCUS_NODES} 'mkdir -p `dirname ${LOCUS_INSTALL_DIR}` && time scp -rq ${LHN_HOST}:${LOCUS_INSTALL_DIR} `dirname ${LOCUS_INSTALL_DIR}` && sudo /sbin/setcap ${STORAGE_CAPABILITIES}=ep ${LOCUS_INSTALL_DIR}/bin/Storage_main'"

# Activate the installation
.PHONY:		activate_bgfen activate_locus
activate_bgfen:		install_ion install_cn install_bgfen
ifneq "${CURRENT_INSTALL}" "${MY_INSTALL}"
	ln -sfT ${CURRENT_INSTALL} ${HOME}/production/lofar-prev
	ln -sfT ${MY_INSTALL} ${HOME}/production/lofar
endif

activate_locus:		install_lhn
ifneq "${CURRENT_INSTALL}" "${MY_INSTALL}"
	ssh ${LHN_HOST} cexec ${LOCUS_NODES} ln -sfT ${LOCUS_HOME}/production/`basename ${CURRENT_INSTALL}` ${LOCUS_HOME}/production/lofar-prev
	ssh ${LHN_HOST} cexec ${LOCUS_NODES} ln -sfT ${LOCUS_HOME}/production/`basename ${MY_INSTALL}` ${LOCUS_HOME}/production/lofar

else        
	ssh ${LHN_HOST} cexec ${LOCUS_NODES} ln -sfT ${LOCUS_HOME}/production/`basename ${PREV_INSTALL}` ${LOCUS_HOME}/production/lofar-prev
	ssh ${LHN_HOST} cexec ${LOCUS_NODES} ln -sfT ${LOCUS_HOME}/production/`basename ${CURRENT_INSTALL}` ${LOCUS_HOME}/production/lofar
endif

# Swap previous and current release
.PHONY:		prev_bgfen prev_locus
prev_bgfen:
	ln -sfT ${CURRENT_INSTALL} ${HOME}/production/lofar-prev
	ln -sfT ${PREV_INSTALL} ${HOME}/production/lofar

prev_locus:
	ssh ${LHN_HOST} cexec ${LOCUS_NODES} ln -sfT ${LOCUS_HOME}/production/`basename ${CURRENT_INSTALL}` ${LOCUS_HOME}/production/lofar-prev
	ssh ${LHN_HOST} cexec ${LOCUS_NODES} ln -sfT ${LOCUS_HOME}/production/`basename ${PREV_INSTALL}` ${LOCUS_HOME}/production/lofar

