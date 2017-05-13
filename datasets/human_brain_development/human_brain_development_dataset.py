

def get_dataset():
    forebrain_file = '../datasets/human_brain_development/forebrain.tsv'

    # Parse data file
    with open(forebrain_file, 'rb') as f:
        tsv_reader = csv.reader(f, delimiter='\t')

        counter = 0
        for row in tsv_reader:
            if counter == 3:
                column_names = row
                prealloc = 100000
                gene_info = []
                expression_data = np.zeros([prealloc, len(column_names[2:])])
                gene_info_column_names = column_names[:2]
                expression_data_column_names = column_names[2:]
                i = 0
            if counter > 3:
                gene_info.append(tuple(row[:2]))  # Save gene info 
                expression_data[i,:] = row[2:]  # Save data to expression data

                i += 1

            counter += 1

    raw_expression_data = expression_data[:i,:]  # Cleave the remainder off

    # Reorder columns
    column_order = [4,5,6,7,8,9,10,11,3,0,1,2]  
    expression_data_column_names = [expression_data_column_names[i] for i in column_order]
    raw_expression_data = raw_expression_data[:,column_order]

    # Determine approximate time point in days
    time_points = np.array([32, 33, 41, 44, 49, 51, 53, 56, 63, 70, 77, 84])

    # Floor bad data
    floor_val = 0.1
    expression_data = np.copy(raw_expression_data)
    expression_data[expression_data<=floor_val] = floor_val

    # Take log of data
    expression_data = np.log2(expression_data)

    # Interpolate 
    intr_time_points = np.arange(32, 85, 1)
    intr_expression_data = [scipy.interpolate.interp1d(time_points, expression_data[i,:])(intr_time_points) 
                            for i in range(expression_data.shape[0])]
    intr_expression_data = np.vstack(intr_expression_data)

    X = np.log2(X)

    mean = np.mean(intr_expression_data, axis=1)
    intr_expression_ms = intr_expression_data - mean[:,np.newaxis]
    expression_ms = expression_data - mean[:,np.newaxis]

    # Filter based on max expression >= t

    #dead_std_thresh = 0.001
    #alive_idx = np.logical_and(np.std(intr_expression_ms, axis=1)>=dead_std_thresh, np.amax(intr_expression_data, axis=1)>= min_log_expression)
    min_log_expression = np.log2(16)
    alive_idx = np.amax(intr_expression_data, axis=1)>= min_log_expression
    print('>>Number of active genes: '+str(np.sum(alive_idx)))

    # Remove dead genes
    intr_expression_ms = intr_expression_ms[alive_idx,:]
    expression_ms = expression_ms[alive_idx,:]
    gene_info_ms = [g for i,g in enumerate(gene_info) if alive_idx[i]]

