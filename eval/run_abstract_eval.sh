for name in academy_curriculum_eb opponents_curriculum_015 opponents_curriculum_0125 checkpoints_selfplay ; do
    echo $name
    ./run.sh --name $name
done
