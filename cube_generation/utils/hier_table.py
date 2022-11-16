def extract_merge_regions(table):
        merged_regions = []
        merged_regions_kv = {}
        for i, row_cells in enumerate(table):
            j = 1
            while j < len(table[0]):
                if row_cells[j] == '':
                    j += 1
                else:
                    first_j = j
                    while j + 1 < len(table[0]) and row_cells[j + 1] == '':
                        j += 1
                    if first_j != j:
                        merged_region = {
                            'FirstRow': i,
                            'LastRow': i,
                            'FirstColumn': first_j,
                            'LastColumn': j
                        }
                        merged_regions.append(merged_region)
                        merged_regions_kv[(i, first_j)] = merged_region
                    j += 1
        i = 0
        while i < len(table):
            if table[i][0] == '':
                i += 1
            else:
                first_i = i
                while i + 1 < len(table) and table[i + 1][0] == '':
                    i += 1
                if first_i != i:
                    merged_region = {
                        'FirstRow': first_i,
                        'LastRow': i,
                        'FirstColumn': 0,
                        'LastColumn': 0
                    }
                    merged_regions.append(merged_region)
                    merged_regions_kv[(first_i, 0)] = merged_region
                i += 1
        return merged_regions, merged_regions_kv



def estimate_header_rows(table, merged_regions_kv, use_num_digits=True, digit_threshold=0.7):
        """ Estimate number of header rows."""
        i = 0
        while i < len(table):
            if table[i][0] != '':
                break
            i += 1
        num_header_rows = max(1, i)
        if use_num_digits:
            for i in range(num_header_rows, len(table)):
                num_chars, num_digits = 0, 0
                for j in range(1, len(table[0])):
                    cell = table[i][j]
                    if cell.lower() == 'n/a':
                        return num_header_rows

                    for char in cell:
                        if char != ' ':
                            num_chars += 1
                        if char.isdigit():
                            num_digits += 1
                if num_digits / (num_chars + 1e-6) < digit_threshold:
                    num_header_rows += 1
                else:
                    break
        return num_header_rows


def estimate_header_columns(table, merged_regions_kv):
        """ Estimate number of header columns."""
        for (i, j) in merged_regions_kv.keys():
            if i != 0 and j == 0:
                return 2
        return 1


def find_row_column_header(table):
    """
    to find header (row & column) of the table
    """
    merged_regions, merged_regions_kv = extract_merge_regions(table)
    num_header_rows = estimate_header_rows(table, merged_regions_kv)
    num_header_cols = estimate_header_columns(table, merged_regions_kv)

    return (num_header_rows, num_header_cols)