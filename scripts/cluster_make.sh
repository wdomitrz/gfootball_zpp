#!/bin/bash


if [ "$#" -lt "3" ]
then    
    echo "name_prefix number_of_machines operation"
    exit
fi

USER="staniszewskiconrad"

RAY_LOCATION="/home/$USER/.local/bin/ray"
RAY_PORT="6012"

NAME_PREFIX="cluster-$1"
NUM_MACHINES=$2

OPERATION=$3

instances=""
for (( i=1; i <= NUM_MACHINES; i++ ))
do
    instance_name="$NAME_PREFIX-$i"
    instances="$instances $instance_name"
    
done

function ssh_exec {
    cmd="$1"
    inst="$2"
    for i in $inst
    do
	echo "$i $cmd"
	gcloud compute ssh "$USER@$i" --command "$cmd"
    done
}

read -r -a array <<< "$instances"
root_machine=${array[0]}
num_workers=$(( NUM_MACHINES - 1 ))
workers=${array[@]:1:NUM_MACHINES}

if [ "$OPERATION" == "create" ]
then
    SNAPSHOT="skonrad-snapshot-cluster-1"
    ZONE='us-central1-a'
    ACCELERATOR='type=nvidia-tesla-t4,count=1'
    MACHINE_TYPE='n1-standard-16'

    echo "Creating $instances"
    gcloud compute instances create $instances \
    	   --source-snapshot $SNAPSHOT \
    	   --zone $ZONE \
    	   --machine-type $MACHINE_TYPE
    
elif [ "$OPERATION" == "fix_horovod" ]
then
    ssh_exec "ssh-keyscan -t rsa,dsa $instances > /home/$USER/.ssh/known_hosts" "$instances"
elif [ "$OPERATION" == "stop" ]
then
    
    gcloud compute instances stop $instances
elif [ "$OPERATION" == "start" ]
then
    gcloud compute instances start $instances
elif [ "$OPERATION" == "delete" ]
then
    gcloud compute instances delete $instances
elif [ "$OPERATION" == "names" ]
then
    echo $instances
elif [ "$OPERATION" == "ray_run" ]
then
    ssh_exec "$RAY_LOCATION start --head --redis-port=$RAY_PORT" \
	     "$root_machine"

    ssh_exec "$RAY_LOCATION start --address=$root_machine:$RAY_PORT" \
	     "$workers"
elif [ "$OPERATION" == "ray_stop" ]
then
    ssh_exec "$RAY_LOCATION stop" $instances
elif [ "$OPERATION" == "ssh" ]
then
    gcloud compute ssh "$USER@$root_machine"
elif [ "$OPERATION" == "fix_firewall" ]
then
    gcloud compute firewall-rules create ssh --allow tcp:22 --source-ranges=0.0.0.0/0 --description="ssh"
    gcloud compute firewall-rules create allow-tensorboard --allow tcp:6006 --source-ranges=0.0.0.0/0 --description="allow-tensorboard"
else
    echo "Bad op"
fi


