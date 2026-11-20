# sourced by every ccpi script: keeps ALL files inside ~/ccpi and ROS 2 traffic on localhost
export CCPI=$HOME/ccpi
export MAMBA_ROOT_PREFIX=$CCPI/mamba CONDARC=$CCPI/condarc MAMBARC=$CCPI/condarc MAMBA_REGISTER_ENVS=false
export XDG_CACHE_HOME=$CCPI/cache XDG_CONFIG_HOME=$CCPI/config XDG_DATA_HOME=$CCPI/data_xdg
export ROS_HOME=$CCPI/roshome ROS_LOG_DIR=$CCPI/roshome/log TMPDIR=$CCPI/tmp
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=1
export PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
