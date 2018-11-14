from amfe.wrappers.ansys_wrapper import read_ansys_sparse_matrix

print('ok')


ansys_K_filename = r'ANSYS\Ksparse.matrix'
ansys_M_filename = r'ANSYS\Msparse.matrix'


K = read_ansys_sparse_matrix(ansys_K_filename)
M = read_ansys_sparse_matrix(ansys_M_filename)

K
M
