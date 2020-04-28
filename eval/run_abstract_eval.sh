for name in academy_curriculum_eb opponents_curriculum_0125 checkpoints_selfplay from0to1to5 scon_e3_p2_hard_sp randomized_env m_sp_in_ob_32_e61_p1 m_fsp_e71_64 checkpoints_sp_c_e1 partially_trained_randomized_diff random random_net ; do
    echo $name
    ./run.sh --name $name
done
