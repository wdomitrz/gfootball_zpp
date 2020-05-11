for name in checkpoints_sp_c_e2 m_fsp_scon_e82_24 checkpoints_sp_c_nb_e2 f0to1to5tosp_e5 m_fsp_e71_64 m_sp_in_ob_32_e61_p1 m_sp_bt_32_e26_p1 checkpoints_selfplay_e9 ; do
    echo $name
    ./run.sh --name $name
done
