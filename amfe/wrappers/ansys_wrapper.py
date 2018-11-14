from scipy import sparse
import numpy as np

def read_ansys_sparse_matrix(filename):
    ''' This function read Ansys sparse matrix file
    
    paramenters:
        filename : str
            string containing the ansys file path
            
    return 
        scipy.sparse.matrix
    '''
    
    with open(filename,'r') as fileobj:
    
        filelines = fileobj.readlines()
    
        last_line = filelines[-1] 
    
        matrix_size = int(last_line.split(',')[0].split('[')[1])
        #print(matrix_size)
        
        S = sparse.dok_matrix((matrix_size, matrix_size), dtype=np.float)
        #S = {}
        for line in  filelines:
            if line[0]=='[':
                for elem_str in line.split('[')[1:]:
                    try:
                        str_ij, float_val = elem_str.split(']:')
                        i,j = str_ij.split(',')
                        S[int(i)-1,int(j)-1] = np.float(float_val)
                    except:
                        print(elem_str)
                        raise('Ansys file format not excepted! Please check the file format!')

            else:
                continue
    
    #row_ind = np.array(list(S.keys())).T[0,:]
    #col_ind = np.array(list(S.keys())).T[1,:]
    #data = np.array(list(S.values()))

    
    return S.tocsc()
    