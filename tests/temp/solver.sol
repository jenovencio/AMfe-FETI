���      �	dill.dill��_create_type���(h �
_load_type����type���R��ParallelSolver�h�object���R���}�(�
__module__��__main__��__init__�h �_create_function���(h�CodeType���R�(KK KKKCCg | _ g | _g | _d S �N���residual��	lampda_im��
lampda_ker����self����7H:\TUM-PC\Dokumente\Projects\AMfe\amfe\MPIfetisolver.py�hM�C �))t�R�c__builtin__
__main__
hNN}�t�R��
mpi_solver�h(h(KK K!KKB�  t |kot dk�r�t atjd� tjdt |f � tj|jt �� |� _� j� }� j	}� j
}tjt||�}xD� jjD ]8}	tjdt|	f � t|	f|krttj|t|	f |	d� qtW i }
x6� jjD ]*}	tjdt|	f � tj|	d�|
|	tf< q�W tj|� tj|
� tjd� tjtj� x*|
D ]"\}}|
|	tf }� j||� �qW � j� }� j
}tjd	� tj|j� � tjtjtjtjg}|tj� jt||fg}ttt||t� tjd
� tjtjj� � tj�  tjdtj  � tjdtj! � tj"� }tj#}� fdd�}dd� }||�}t$� tt�}|| }tj%|||�\}}}}|| }tj&� }|||� }tj'|�\}}� j(|tj)d� � j*|tj+� t,j-j.t/t0t�d �} tj1� | � � S tjdt  � dS dS �(�N solve linear FETI problem with PCGP with parcial reorthogonalization
        �K �6%%%%%%%%%%%%%%%%%%% START %%%%%%%%%%%%%%%%%%%%%%%%%%%%��Solving domain %i from size %i��'
Sending message from %i to neighbor %i��dest����3
Receiving message at subdomain %i from neighbor %i��source����G_dict��Local GtG rows��Master GtG keys��total interface dof %i��Null space size %i�h(KK KKKCt | � tt�S �N���action_of_global_F_mpi��master��partitions_list����x���h�<lambda>�M�C ��sub_i���)t�R��+ParallelSolver.mpi_solver.<locals>.<lambda>�h(KK KKKSC
t | t�S �N���projection_action_mpi�h8���rk���hh=M�C �))t�R��global_lambda��lambda_dict����.are��4%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%��!Nothing to do on from process %i �Nt�(�rank��sub_id��logging��info��amfe��FETIsubdomain��groups��set_cholesky_tolerance��calc_G_dict��num_of_interface_dof_dict��null_space_size�h8�append_partition_dof_info_dicts��submesh��neighbor_partitions��comm��send��debug��recv��append_G_dict�h0�append_neighbor_G_dict��calc_GtG_row��keys��appendGtG_row��append_null_space_force��append_partition_tuple_info��null_space_force��exchange_info�h9�GtG_row_dict��build_local_to_global_mapping��total_interface_dof��total_nullspace_dof��solve_lambda_im��
lambda_ker��assemble_global_d_mpi��PCGP��assemble_global_d_hat��solve_corse_grid��solve_local_displacement�hN�apply_rigid_body_correction��
alpha_dict��os��path��join��	directory��str��save_object�t�(h�
sub_domain��num_partitions��n_int��cholesky_tolerance��Gi_dict��subdomain_interface_dofs_dict��subdomain_null_space_size��local_info_dict��nei_id��Gj_dict��nei_key��sub_key��Gj��GtG_rows_dict�h^�master_func_list��var_list��	lambda_im�ht�F��P��Fim��d��r0��last_res��proj_r_hist��lambda_hist��
lambda_sol��d_hat��wk��global_alpha��res_path�t�hh%M�C� 





 �)h?��t�R�c__builtin__
__main__
h%M�G>�����퍆�N}�t�R��__doc__�N�__slotnames__�]�ut�R�)��}�(h]�h]�h]�ub.