��      �	dill.dill��_create_type���(h �
_load_type����type���R��ParallelSolver�h�object���R���}�(�
__module__��__main__��__init__�h �_create_function���(h�CodeType���R�(KK KKKCC
g | _ d S �N���residual����self����7H:\TUM-PC\Dokumente\Projects\AMfe\amfe\MPIfetisolver.py�hK�C �))t�R�c__builtin__
__main__
hNN}�t�R��
mpi_solver�h(h(KK K,KKCB�  t |kot dk�r�t atjd� tjdt |f � tj|jt �}||_|j� }tjdt	t�d|j
� � xD|jjD ]8}tjdt|f � t|f|krrtj|t|f |d� qrW tt_tj|t� i }xB|jjD ]6}tjdt|f � tj|d	�||tf< tj||� q�W tj|j
� � t||t�}	|j}
tjd
� tj|	j
� � tjtjtjg}|	|j|jg}ttt||t� tjd� tjtjj
� � tjd� x tjD ]}tjtj| � �q�W tjd� tjtjj
� � x$tjD ]}tjtj| j� � �q�W tjd� tjtjj
� � x tjD ]}tjtj| � �qW tjdtj  � tjdtj! � tjdtj" � tjdtj# � tj$� }tj%}tj&}tjd� tjtj%� tjd� tj|� tj'}tjd� tj|� t(|||�}tjd� tj|j
� � x|D ]}tj|| � �q�W tj)tj*g}||j+g}ttt||t� tj,� }tjd� tj|� tj-� }tjd� tj|� t.|�}t/j0||�}i }i }t/j1|dg�}|| }|}�x�t2|�D �]�}t/j3j4|�}tjd||f � tj5|�\}}t/j3j4|�} | j6j7| � tjd|| f � | |k �rP |j8||� tjd� tj|j9� t/j:||�}!|dk�r||d j;� |d< |d j;� |d< ||d< |!|d< n||d< |!|d< t<|||�}"tjd|"� |!|"|  }#tjd|#j=� t(|||#�}tj)tj*g}||j+g}ttt||t� tj,� }$tjd|$j=� t>|!||#|$�}%tjd|%� |%|$ }&||%|#  }||& }|#}�q�W || }'tjd � tj|'j=� tj-� }t?t|||'�}(tjd!|(j=� |})||( })tj5|)�\}}t@||'|�}*tjd"|*j=� tAjBjCtDt	t�d# �}+tjE||+� |S tjd%t  � d&S d&S �(�N solve linear FETI problem with PCGP with parcial reorthogonalization
        �K �6%%%%%%%%%%%%%%%%%%% START %%%%%%%%%%%%%%%%%%%%%%%%%%%%��Solving domain %i from size %i��Domain��G_dict =��'
Sending message from %i to neighbor %i��dest����3
Receiving message at subdomain %i from neighbor %i��source����Local GtG rows��Master GtG keys��Local rows for each subdomain��B dict��G dict��total interface dof %i��total dof %i��Null space size %i��master.lambda_im��lambda_im_dict��lambda_id_dict��local_h_dict.keys()��Fim��d�K�iteration %i, norm rk = %f�� iteration %i, norm wk = %0.2e /n��u_bar��beta = ��pk��h = ��alpha =��
lambda_sol��
F_lambda =��u =��.are��4%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%��!Nothing to do on from process %i �Nt�(�rank��sub_id��logging��debug��amfe��FETIsubdomain��groups��set_cholesky_tolerance��calc_G_dict��str��keys��submesh��neighbor_partitions��comm��send��partitions_list��master��subdomain_keys��appendG��recv��create_GtG_rows��null_space_size��appendGtG_row��append_local_B��append_null_space_force��B_dict��null_space_force��exchange_info��GtG_row_dict��Bi_dict��todense��G_dict��total_interface_dof��	total_dof��total_nullspace_dof��course_grid_size��solve_lambda_im��	lambda_im��
lambda_ker�h:�subdomain_apply_F��append_h��append_d_hat��dual_force_dict��
assemble_h��assemble_global_d_hat��len��np��eye��zeros��range��linalg��norm��solve_corse_grid�h�append��solve_local_displacement�h@�matmul��copy��	beta_calc��T��
alpha_calc��global_apply_F��subdomain_step4��os��path��join��	directory��save_object�t�(h�
sub_domain��num_partitions��n_int��tol��sub_i��Gi_dict��nei_id��Gj_dict��GtG_rows_dict�ha�master_func_list��var_list��key�h9hqhrh:�local_h_dict�h<h=�n��precond��w_dict��y_dict��pk1��r0��rk��i��norm_rk��wk��	alpha_hat��norm_wk��yk��beta�hB�h��alpha��delta_rk�hE�F_lambda��d_hat��u��res_path�t�hh#MB   
























 �))t�R�c__builtin__
__main__
h#M�G>�����퍆�N}�t�R��__doc__�N�__slotnames__�]�ut�R�)��}�h]�h �	_get_attr���h �_import_module����numpy.core.multiarray���R��scalar���R��numpy��dtype����f8�K K��R�(K�<�NNNJ����J����K t�bCC!����>���R�asb.