#!/bin/bash

if [ "$#" -ne 4 ]
then
    echo "OPERATION(user_setup,enter), MACHINE, USER, ZONE"
    exit 1
fi

OPERATION="$1"
MACHINE="$2"
USER="$3"
ZONE="$4"
PROJECT="warsaw-zpp"

function fix_ssh {
    gcloud compute firewall-rules create ssh --allow tcp:22 --source-ranges=0.0.0.0/0 --description="ssh"
}

function ssh_exec {
    local cmd="$1"
    gcloud compute ssh "$USER@${MACHINE}" --command "$cmd" --zone="$ZONE"
}
function ssh_machine {
    gcloud compute ssh "$USER@${MACHINE}" --zone="$ZONE"
}

function push_folder {
    local fpath="$1"
    gcloud compute scp "$fpath" "$USER@${MACHINE}:~" --zone="$ZONE"
}

function mount_home {
    local ldir="$1"
    mkdir -p "$1"
    sshfs -o IdentityFile=~/.ssh/google_compute_engine "${USER}@${MACHINE}.${ZONE}.${PROJECT}:" "$ldir"
}

function prepare_machine {
    ssh_exec "$(cat remote_setup.sh)"
}

if [ "$OPERATION" == "user_setup" ]
then
    ssh_exec "sudo usermod -aG docker $USER;"
    gcloud compute config-ssh
    mount_folder="./${MACHINE}_home"
    mount_home "$mount_folder"
    cd "$mount_folder"
    git clone https://github.com/CStanKonrad/seed_rl.git
    cd seed_rl
    git checkout current_setup
    cd ../
    git clone https://github.com/wdomitrz/gfootball_zpp.git
    cd ..
    fusermount -u "$mount_folder"
elif [ "$OPERATION" == "enter" ]
then
    gcloud compute config-ssh
    mount_home "${MACHINE}_home"
    ssh_machine
fi
