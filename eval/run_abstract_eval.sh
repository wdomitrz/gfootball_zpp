for name in academy_curriculum_eb opponents_curriculum_015 opponents_curriculum_0125 checkpoints_selfplay m_sp_in_bt_4_e2_p1 ; do
    echo $name
    ./run.sh --name $name
done
