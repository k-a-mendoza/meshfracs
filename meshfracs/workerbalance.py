import numpy as np

def create_worker_patches(worker_div_multiplier, worker_count, d2_grid_shape, x_grid, y_grid):

    worker_count_2 = int(np.floor(np.sqrt(2*worker_count))*worker_div_multiplier)

    index_ranges, patch_size = _create_base_divisions(d2_grid_shape, x_grid, y_grid, worker_count_2)
    index_ranges = _subdivide_patches_below_limit(index_ranges, patch_size, x_grid, y_grid)

    return index_ranges


def _divide_patch(index_patch, x_grid, y_grid, limit=100):
    ix0, ix1 = index_patch[0][0], index_patch[0][1]
    iy0, iy1 = index_patch[1][0], index_patch[1][1]

    x_divisible = (ix1 - ix0)/2 > 1
    y_divisible = (iy1 - iy0)/2 > 1

    divide_x = x_divisible > y_divisible

    if not x_divisible and not y_divisible:
        return [index_patch], [limit-0.1]

    elif y_divisible and not x_divisible:
        divide_x=False

    elif not y_divisible and x_divisible:
        divide_x=True

    if divide_x:
        ix_mid = int((ix1+ix0)//2)
        patch_1 = ((ix0, ix_mid), (iy0,iy1))
        patch_2 = ((ix_mid, ix1), (iy0,iy1))
    else:
        iy_mid = int((iy1+iy0)//2)
        patch_1 = ((ix0, ix1), (iy0, iy_mid))
        patch_2 = ((ix0, ix1), (iy_mid, iy1))

    dx1 = np.squeeze(x_grid[patch_1[0][1],0] - x_grid[patch_1[0][0],0])
    dx2 = np.squeeze(x_grid[patch_2[0][1],0] - x_grid[patch_2[0][0],0])
    dy1 = np.squeeze(y_grid[0,patch_1[1][1]] - y_grid[0,patch_1[1][0]])
    dy2 = np.squeeze(y_grid[0,patch_2[1][1]] - y_grid[0,patch_2[1][0]])
    return [patch_1] + [patch_2], [dx1*dy1, dy2*dx2]

        
def _create_base_divisions(d2_grid_shape, x_grid, y_grid, worker_count_2):
    x_indices = np.arange(0,d2_grid_shape[0]-1)
    y_indices = np.arange(0,d2_grid_shape[1]-1)

    x_index_split = np.array_split(x_indices,worker_count_2)
    y_index_split = np.array_split(y_indices,worker_count_2)

    index_ranges = []
    patch_size   = []
    for valid_indices_x in x_index_split:
        ix_min = np.amin(valid_indices_x)
        ix_max = np.amax(valid_indices_x)+1
        delta_x = np.squeeze(x_grid[ix_max,0]-x_grid[ix_min,0])

        for valid_indices_y in y_index_split:
            iy_min = np.amin(valid_indices_y)
            iy_max = np.amax(valid_indices_y)+1
            delta_y = np.squeeze(y_grid[0,iy_max]-y_grid[0,iy_min])
            patch_size.append(delta_x*delta_y)

            index_ranges.append(((ix_min, ix_max), (iy_min, iy_max)))

    print(f'initial work division is {len(index_ranges)} sections')
    return index_ranges, patch_size

def _subdivide_patches_below_limit(index_ranges, patch_size, x_grid, y_grid,area_log_cutoff=12):
    size_limit = np.power(10,area_log_cutoff)
    print('optimizing patches')
    iterable_index  = []
    new_patch_sizes = []
    while len(patch_size)>0:
        index_patch = index_ranges.pop()
        area = patch_size.pop()
        if area < size_limit:
            iterable_index.append(index_patch)
            new_patch_sizes.append(area)
        else:
            new_indices, new_patches = _divide_patch(index_patch, x_grid, y_grid,limit=size_limit)
            index_ranges.extend(new_indices)
            patch_size.extend(new_patches)


    new_patch_sizes, iterable_index = zip(*sorted(zip(new_patch_sizes, iterable_index),reverse=True))
    print(f'optimized work division is {len(iterable_index)} sections')
    return iterable_index
