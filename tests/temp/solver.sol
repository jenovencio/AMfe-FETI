��^      �	dill.dill��_create_type���(h �
_load_type����type���R��ParallelSolver�h�object���R���}�(�
__module__��__main__��__init__�h �_create_function���(h�CodeType���R�(KK KKKCC
g | _ d S �N���residual����self����7H:\TUM-PC\Dokumente\Projects\AMfe\amfe\MPIfetisolver.py�hM	C �))t�R�c__builtin__
__main__
hNN}�t�R��
mpi_solver�h(h(KK K,KKCB|  t |kot dk�rht atd� tdt |f � tt�\}}t|j� � xB|jjD ]6}tdt|f � t|f|krPtj|t|f |d� qPW t	t
_t
j|t� i }x@|jjD ]4}tdt|f � tj|d�||tf< t
j||� q�W t|j� � t||t�}	|j}
td� t|	j� � t
jt
jt
jg}|	|j|jg}ttt
||t	� td	� tt
jj� � td
� xt
jD ]}tt
j| � �qhW td� tt
jj� � x"t
jD ]}tt
j| j� � �q�W td� tt
jj� � xt
jD ]}tt
j| � �q�W tdt
j � tdt
j � tdt
j � tdt
j � t
j� }t
j}t
j }td� tt
j� td� t|� t
j!}td� t|� t"|||�}td� t|j� � x|D ]}t|| � �q�W t
j#t
j$g}||j%g}ttt
||t	� t
j&� }td� t|� t
j'� }td� t|� t(|�}t)j*||�}i }i }t)j+|dg�}|| }|}�x�t,|�D �]�}t)j-j.|�}td||f � t
j/|�\}}t)j-j.|�} | j0j1| � td|| f � | |k �r�P |j2||� td� t|j3� t)j4||�}!|dk�r|d j5� |d< |d j5� |d< ||d< |!|d< n||d< |!|d< t6|||�}"td|"� |!|"|  }#td|#j7� t"|||#�}t
j#t
j$g}||j%g}ttt
||t	� t
j&� }$td|$j7� t8|!||#|$�}%td|%� |%|$ }&||%|#  }||& }|#}�qHW || }'td� t|'j7� t
j'� }t9t
|||'�}(td|(j7� |})||( })t
j/|)�\}}t:||'|�}*td |*j7� t;j<j=t>t?t�d! �}+t@jA||+� |S td#t  � d$S d$S �(�N solve linear FETI problem with PCGP with parcial reorthogonalization
        �K �6%%%%%%%%%%%%%%%%%%% START %%%%%%%%%%%%%%%%%%%%%%%%%%%%��Solving domain %i from size %i��'
Sending message from %i to neighbor %i��dest����3
Receiving message at subdomain %i from neighbor %i��source����Local GtG rows��Master GtG keys��Local rows for each subdomain��B dict��G dict��total interface dof %i��total dof %i��Null space size %i��master.lambda_im��lambda_im_dict��lambda_id_dict��local_h_dict.keys()��Fim��d�K�iteration %i, norm rk = %f�� iteration %i, norm wk = %0.2e /n��u_bar��beta = ��pk��h = ��alpha =��
lambda_sol��
F_lambda =��u =��.are��4%%%%%%%%%%%%%%%%%%% END %%%%%%%%%%%%%%%%%%%%%%%%%%%%��!Nothing to do on from process %i �Nt�(�rank��sub_id��print��subdomain_i��keys��submesh��neighbor_partitions��comm��send��partitions_list��master��subdomain_keys��appendG��recv��create_GtG_rows��null_space_size��appendGtG_row��append_local_B��append_null_space_force��B_dict��null_space_force��exchange_info��GtG_row_dict��Bi_dict��todense��G_dict��total_interface_dof��	total_dof��total_nullspace_dof��course_grid_size��solve_lambda_im��	lambda_im��
lambda_ker�h8�subdomain_apply_F��append_h��append_d_hat��dual_force_dict��
assemble_h��assemble_global_d_hat��len��np��eye��zeros��range��linalg��norm��solve_corse_grid�h�append��solve_local_displacement�h>�matmul��copy��	beta_calc��T��
alpha_calc��global_apply_F��subdomain_step4��os��path��join��	directory��str��amfe��save_object�t�(h�
sub_domain��num_partitions��n_int��tol��sub_i��Gi_dict��nei_id��Gj_dict��GtG_rows_dict�hY�master_func_list��var_list��key�h7hihjh8�local_h_dict�h:h;�n��precond��w_dict��y_dict��pk1��r0��rk��i��norm_rk��wk��	alpha_hat��norm_wk��yk��beta�h@�h��alpha��delta_rk�hC�F_lambda��d_hat��u��res_path�t�hh#MC� 











 �))t�R�c__builtin__
__main__
h#M�G=�|��׽���N}�t�R��__doc__�N�__slotnames__�]�ut�R�)��}�h]�(h �	_get_attr���h �_import_module����numpy.core.multiarray���R��scalar���R��numpy��dtype����f8�K K��R�(K�<�NNNJ����J����K t�bCC!����>���R�h�h�C��ci3Պ;���R�esb.