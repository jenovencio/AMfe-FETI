import collections

class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = [] 
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:        
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

def get_dofs(submesh_obj, direction ='xyz', id_matrix=None):
    ''' get dofs given a submesh and a global id_matrix
    
    # parameters:
        # submesh_obj : amfe.SubMesh
            # submesh object with nodes and element of dirichlet
        # direction : str
            # direction to consirer 'xyz'
        # id_matrix : dict
            # dict maps nodes to DOFs
            
    # return 
        # dir_dofs : list
            # list with Dirichlet dofs
    '''
 
    x_dir = 0
    y_dir = 1
    z_dir = 2
    
    dofs_to_keep = []
    if 'x' in direction:
        dofs_to_keep.append(x_dir)

    if 'y' in direction:
        dofs_to_keep.append(y_dir)
    
    if 'z' in direction:
        dofs_to_keep.append(z_dir)
    
    dir_nodes = submesh_obj.global_node_list
    dir_dofs = []
    for node in dir_nodes:
        dofs = id_matrix[node]
        local_dofs = []
        for i in dofs_to_keep:
            try:
                local_dofs.append(dofs[i])
            except:
                print('It is not possible to find dof %i as dirichlet dof' %dofs[i])
        dir_dofs.extend(local_dofs)
    
    return dir_dofs

def create_dof_to_node_map(id_matrix):
    ''' id matrix has x, y, and z in the sequence
    
    '''
    dof_to_node = {}
    dof_to_direction = {}
    direction_list = ['x','y','z']
    for node_id, dof_list in id_matrix.items():
        for direction_id, dof_id in enumerate(dof_list):
            
            dof_to_node[dof_id] = node_id
            dof_to_direction[dof_id] = direction_list[direction_id]
            
    return dof_to_node, dof_to_direction

        
if __name__ == '__main__':
    s = OrderedSet('abracadaba')
    t = OrderedSet('simsalabim')
    print(s | t)
    print(s & t)
    print(s - t)