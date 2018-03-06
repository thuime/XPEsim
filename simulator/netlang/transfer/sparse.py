import numpy


def block_sparse(x, block_size, pruning_rate):
    rows, cols = x.shape
    block_rows = (rows - 1) / block_size + 1
    block_cols = (cols - 1) / block_size + 1

    num_pruned = int(block_rows * block_cols * pruning_rate)

    # Create a new x that can be divided into full blocks
    rows_ = block_size * block_rows
    cols_ = block_size * block_cols
    x_ = numpy.zeros(shape=(rows_, cols_))
    x_[:rows, :cols] = numpy.abs(x)

    row_order = numpy.arange(rows_)
    col_order = numpy.arange(cols_)

    prev_hole_sum = numpy.sum(numpy.abs(x_))

    # EM iteration
    while True:
        # E step, choose `num_pruned` blocks as hole
        block_sum = numpy.zeros(shape=(block_rows, block_cols))
        for i in xrange(block_rows):
            for j in xrange(block_cols):
                block_sum[i, j] = numpy.sum(
                    x_[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size])

        block_hole = numpy.zeros(shape=(block_rows, block_cols), dtype=bool)
        for i, j in zip(*(numpy.unravel_index(block_sum.argsort(axis=None)[:num_pruned], dims=block_sum.shape))):
            block_hole[i, j] = True

        # M step
        # row first
        row_hole = block_hole.any(axis=1)
        row_sum = x_.sum(axis=1)
        row_sort = row_sum.argsort()

        hole_start = 0
        remain_start = row_hole.sum() * block_size

        row_order_ = numpy.zeros(shape=(rows_,), dtype=int)
        for i, has_hole in enumerate(row_hole):
            if has_hole:
                row_order_[i * block_size:(i + 1) * block_size] = row_sort[hole_start:hole_start + block_size]
                hole_start += block_size
            else:
                row_order_[i * block_size:(i + 1) * block_size] = row_sort[remain_start:remain_start + block_size]
                remain_start += block_size

        x_ = x_[row_order_, :]
        row_order = row_order[row_order_]

        # col next
        col_hole = block_hole.any(axis=0)
        col_sum = x_.sum(axis=0)
        col_sort = col_sum.argsort()

        hole_start = 0
        remain_start = col_hole.sum() * block_size

        col_order_ = numpy.zeros(shape=(cols_,), dtype=int)
        for i, has_hole in enumerate(col_hole):
            if has_hole:
                col_order_[i * block_size:(i + 1) * block_size] = col_sort[hole_start:hole_start + block_size]
                hole_start += block_size
            else:
                col_order_[i * block_size:(i + 1) * block_size] = col_sort[remain_start:remain_start + block_size]
                remain_start += block_size

        x_ = x_[:, col_order_]
        col_order = col_order[col_order_]

        hole_sum = 0
        for i in xrange(block_rows):
            for j in xrange(block_cols):
                if block_hole[i, j]:
                    hole_sum += x_[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size].sum()

        if hole_sum >= prev_hole_sum:
            break
        else:
            prev_hole_sum = hole_sum

    mask = numpy.zeros(shape=x.shape, dtype=bool)
    for i in xrange(block_rows):
        for j in xrange(block_cols):
            mask[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = not block_hole[i, j]

    row_order = filter(lambda x: x < rows, row_order)
    col_order = filter(lambda x: x < cols, col_order)

    row_swap = [-1] * len(row_order)
    for i, v in enumerate(row_order):
        row_swap[v] = i

    col_swap = [-1] * len(col_order)
    for i, v in enumerate(col_order):
        col_swap[v] = i

    mask = mask[row_swap, :][:, col_swap]

    return mask, row_order, col_order
