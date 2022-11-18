# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import json
import tqdm
import argparse
import numpy as np

from generator.tatqa_generator import TaTQAGenerator
from utils.fuzz_retrieve import get_order_by_tf_idf
from utils.hier_table import find_row_column_header
from utils.tatqa import AVGs, DIFFs, SUMs, DIVIDEs, CHANGE_RATIOs

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./dataset/tatqa/tatqa_dataset_dev.json")
    parser.add_argument("--output_dir", type=str, default="./dataset/tatqa/")
    parser.add_argument("--output_file", type=str, default="tatqa_dev_cube.json")

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    f_dataset = open(args.input_path, "r")
    dataset_result = json.load(f_dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_file)
    f_out_path = open(output_path, "w")
    
    
    data_out = []
    TOP_K = None
    TOP_K_1 = None
    table_cube_generator = TaTQAGenerator()

    for data_item in tqdm.tqdm(dataset_result):

        table = data_item["table"]["table"]
        questions = data_item["questions"]
        data_item_new = data_item.copy()
        header_row_num, header_col_num = find_row_column_header(table.copy())

        for iquestion, question in enumerate(questions):
            q_text = question["question"]
            q_answer = question["answer"]

            _, sim_avg = get_order_by_tf_idf(q_text, AVGs)
            _, sim_diff = get_order_by_tf_idf(q_text, DIFFs)
            _, sim_sums = get_order_by_tf_idf(q_text, SUMs)
            _, sim_divides = get_order_by_tf_idf(q_text, DIVIDEs)
            _, sim_change_ratio = get_order_by_tf_idf(q_text, CHANGE_RATIOs)

            sim_avg = max([sim[1] for sim in sim_avg])
            sim_diff = max([sim[1] for sim in sim_diff])
            sim_sums = max([sim[1] for sim in sim_sums])
            sim_divides = max([sim[1] for sim in sim_divides])
            sim_change_ratio = max([sim[1] for sim in sim_change_ratio])

            sim_list = [
                sim_avg,
                sim_diff,
                sim_sums,
                sim_divides,
                sim_change_ratio,
            ]

            #######################
            #######################
            # extract candidate rows / columns according to question
            (
                candidate_rows,
                candidate_cols,
                candidate_rows_header_col,
                candidate_cols_header_row,
            ) = table_cube_generator.get_candidate_row_col_from_table(
                q_text=q_text,
                table=table,
                header_row_num=header_row_num,
                header_col_num=header_col_num,
            )

            # if extract only one header, then use other header's row/col as backup
            if len(candidate_rows) == 0:
                candidate_rows = candidate_rows_header_col
            elif len(candidate_cols) == 0:
                candidate_cols = candidate_cols_header_row

            # total result for datacube: OP, CELL, PSEUDO_ANSWER, ROW_HEADER, COL_HEADER
            op_list, cell_list, num_list, row_header_list, col_header_list, ans_list = (
                [],
                [],
                [],
                [],
                [],
                [],
            )

            #######################
            #######################
            # AVG handle
            if sim_avg > 0:
                num_list = []
                (
                    tmp_op_list,
                    tmp_cell_list,
                    tmp_num_list,
                    tmp_ans_list,
                    tmp_row_header_list,
                    tmp_col_header_list,
                ) = ([], [], [], [], [], [])
            
                (
                    cube_op_list,
                    cube_cell_list,
                    cube_num_list,
                    cube_row_header_list,
                    cube_col_header_list,
                ) = table_cube_generator.get_same_row_cube_pattern(
                    OP="AVG",
                    table=table,
                    header_col_num=header_col_num,
                    header_row_num=header_row_num,
                    candidate_rows=candidate_rows,
                    candidate_cols=candidate_cols,
                )
                tmp_op_list += cube_op_list
                tmp_cell_list += cube_cell_list
                tmp_num_list += cube_num_list
                tmp_row_header_list += cube_row_header_list
                tmp_col_header_list += cube_col_header_list

                if (
                    len(candidate_cols) == 2
                    and abs(candidate_cols[0] - candidate_cols[1]) > 1
                ):

                    (
                        cube_op_list,
                        cube_cell_list,
                        cube_num_list,
                        cube_row_header_list,
                        cube_col_header_list,
                    ) = table_cube_generator.get_same_row_cube_pattern(
                        OP="AVG",
                        table=table,
                        header_col_num=header_col_num,
                        header_row_num=header_row_num,
                        candidate_rows=candidate_rows,
                        candidate_cols=[
                            icol
                            for icol in range(candidate_cols[0], candidate_cols[1] + 1)
                        ],
                    )
                    tmp_op_list += cube_op_list
                    tmp_cell_list += cube_cell_list
                    tmp_num_list += cube_num_list
                    tmp_row_header_list += cube_row_header_list
                    tmp_col_header_list += cube_col_header_list

                if len(candidate_cols) < len(table[0]):

                    (
                        cube_op_list,
                        cube_cell_list,
                        cube_num_list,
                        cube_row_header_list,
                        cube_col_header_list,
                    ) = table_cube_generator.get_same_row_cube_pattern(
                        OP="AVG",
                        table=table,
                        header_col_num=header_col_num,
                        header_row_num=header_row_num,
                        candidate_rows=candidate_rows,
                        candidate_cols=candidate_cols,
                        all_flag=True,
                    )
                    tmp_op_list += cube_op_list
                    tmp_cell_list += cube_cell_list
                    tmp_num_list += cube_num_list
                    tmp_row_header_list += cube_row_header_list
                    tmp_col_header_list += cube_col_header_list


                avg_list = []
                for item in tmp_num_list:
                    avg_list.append(round(np.mean(item), 2))
                    tmp_ans_list.append(round(np.mean(item), 2))

                (
                    tmp_op_list,
                    tmp_cell_list,
                    tmp_num_list,
                    tmp_ans_list,
                    tmp_row_header_list,
                    tmp_col_header_list,
                ) = table_cube_generator.generate_ranked_cube_result(
                    q_text=q_text,
                    op_list=tmp_op_list,
                    cell_list=tmp_cell_list,
                    num_list=tmp_num_list,
                    ans_list=tmp_ans_list,
                    row_header_list=tmp_row_header_list,
                    col_header_list=tmp_col_header_list,
                    top_k=TOP_K if sim_avg != max(sim_list) else TOP_K_1,
                )

                op_list += tmp_op_list
                cell_list += tmp_cell_list
                num_list += tmp_num_list
                ans_list += tmp_ans_list
                row_header_list += tmp_row_header_list
                col_header_list += tmp_col_header_list

            #######################
            #######################
            # SUM handle
            if sim_sums > 0:
                num_list = []

                (
                    tmp_op_list,
                    tmp_cell_list,
                    tmp_num_list,
                    tmp_ans_list,
                    tmp_row_header_list,
                    tmp_col_header_list,
                ) = ([], [], [], [], [], [])

                (
                    cube_op_list,
                    cube_cell_list,
                    cube_num_list,
                    cube_row_header_list,
                    cube_col_header_list,
                ) = table_cube_generator.get_same_row_cube_pattern(
                    OP="SUM",
                    table=table,
                    header_col_num=header_col_num,
                    header_row_num=header_row_num,
                    candidate_rows=candidate_rows,
                    candidate_cols=candidate_cols,
                )
                tmp_op_list += cube_op_list
                tmp_cell_list += cube_cell_list
                tmp_num_list += cube_num_list
                tmp_row_header_list += cube_row_header_list
                tmp_col_header_list += cube_col_header_list

                (
                    cube_op_list,
                    cube_cell_list,
                    cube_num_list,
                    cube_row_header_list,
                    cube_col_header_list,
                ) = table_cube_generator.get_same_col_cube_pattern(
                    OP="SUM",
                    table=table,
                    header_col_num=header_col_num,
                    header_row_num=header_row_num,
                    candidate_rows=candidate_rows,
                    candidate_cols=candidate_cols,
                )
                tmp_op_list += cube_op_list
                tmp_cell_list += cube_cell_list
                tmp_num_list += cube_num_list
                tmp_row_header_list += cube_row_header_list
                tmp_col_header_list += cube_col_header_list

                if len(candidate_cols) < len(table[0]):
                    (
                        cube_op_list,
                        cube_cell_list,
                        cube_num_list,
                        cube_row_header_list,
                        cube_col_header_list,
                    ) = table_cube_generator.get_same_row_cube_pattern(
                        OP="SUM",
                        table=table,
                        header_col_num=header_col_num,
                        header_row_num=header_row_num,
                        candidate_rows=candidate_rows,
                        candidate_cols=candidate_cols,
                        all_flag=True,
                    )
                    tmp_op_list += cube_op_list
                    tmp_cell_list += cube_cell_list
                    tmp_num_list += cube_num_list
                    tmp_row_header_list += cube_row_header_list
                    tmp_col_header_list += cube_col_header_list

                sum_list = []
                for item in tmp_num_list:
                    sum_list.append(round(np.sum(item), 2))
                    tmp_ans_list.append(round(np.sum(item), 2))

                (
                    tmp_op_list,
                    tmp_cell_list,
                    tmp_num_list,
                    tmp_ans_list,
                    tmp_row_header_list,
                    tmp_col_header_list,
                ) = table_cube_generator.generate_ranked_cube_result(
                    q_text=q_text,
                    op_list=tmp_op_list,
                    cell_list=tmp_cell_list,
                    num_list=tmp_num_list,
                    ans_list=tmp_ans_list,
                    row_header_list=tmp_row_header_list,
                    col_header_list=tmp_col_header_list,
                    top_k=TOP_K if sim_sums != max(sim_list) else TOP_K_1,
                )

                op_list += tmp_op_list
                cell_list += tmp_cell_list
                num_list += tmp_num_list
                ans_list += tmp_ans_list
                row_header_list += tmp_row_header_list
                col_header_list += tmp_col_header_list

            #######################
            #######################
            # Difference handle
            if sim_diff > 0:
                num_list = []

                (
                    tmp_op_list,
                    tmp_cell_list,
                    tmp_num_list,
                    tmp_ans_list,
                    tmp_row_header_list,
                    tmp_col_header_list,
                ) = ([], [], [], [], [], [])

                (
                    cube_op_list,
                    cube_cell_list,
                    cube_num_list,
                    cube_row_header_list,
                    cube_col_header_list,
                ) = table_cube_generator.get_same_row_cube_pattern(
                    OP="DIFF",
                    table=table,
                    header_col_num=header_col_num,
                    header_row_num=header_row_num,
                    candidate_rows=candidate_rows,
                    candidate_cols=candidate_cols,
                )
                tmp_op_list += cube_op_list
                tmp_cell_list += cube_cell_list
                tmp_num_list += cube_num_list
                tmp_row_header_list += cube_row_header_list
                tmp_col_header_list += cube_col_header_list

                
                (
                    cube_op_list,
                    cube_cell_list,
                    cube_num_list,
                    cube_row_header_list,
                    cube_col_header_list,
                ) = table_cube_generator.get_same_col_cube_pattern(
                    OP="DIFF",
                    table=table,
                    header_col_num=header_col_num,
                    header_row_num=header_row_num,
                    candidate_rows=candidate_rows,
                    candidate_cols=candidate_cols,
                )
                tmp_op_list += cube_op_list
                tmp_cell_list += cube_cell_list
                tmp_num_list += cube_num_list
                tmp_row_header_list += cube_row_header_list
                tmp_col_header_list += cube_col_header_list

                (
                    cube_op_list,
                    cube_cell_list,
                    cube_num_list,
                    cube_row_header_list,
                    cube_col_header_list,
                ) = table_cube_generator.get_same_row_cube_pattern(
                    OP="DIFF",
                    table=table,
                    header_col_num=header_col_num,
                    header_row_num=header_row_num,
                    candidate_rows=candidate_rows,
                    candidate_cols=candidate_cols,
                    all_flag=True,
                )
                tmp_op_list += cube_op_list
                tmp_cell_list += cube_cell_list
                tmp_num_list += cube_num_list
                tmp_row_header_list += cube_row_header_list
                tmp_col_header_list += cube_col_header_list

                diff_list = []
                for item in tmp_num_list:
                    if len(item) == 2:
                        diff_list.append(round(item[0] - item[1], 2))
                        tmp_ans_list.append(round(item[0] - item[1], 2))

                (
                    tmp_op_list,
                    tmp_cell_list,
                    tmp_num_list,
                    tmp_ans_list,
                    tmp_row_header_list,
                    tmp_col_header_list,
                ) = table_cube_generator.generate_ranked_cube_result(
                    q_text=q_text,
                    op_list=tmp_op_list,
                    cell_list=tmp_cell_list,
                    num_list=tmp_num_list,
                    ans_list=tmp_ans_list,
                    row_header_list=tmp_row_header_list,
                    col_header_list=tmp_col_header_list,
                    top_k=TOP_K if sim_diff != max(sim_list) else TOP_K_1,
                )

                op_list += tmp_op_list
                cell_list += tmp_cell_list
                num_list += tmp_num_list
                ans_list += tmp_ans_list
                row_header_list += tmp_row_header_list
                col_header_list += tmp_col_header_list

            #######################
            #######################
            # Change Ratio handle
            if sim_change_ratio > 0: 
                num_list = []

                (
                    tmp_op_list,
                    tmp_cell_list,
                    tmp_num_list,
                    tmp_ans_list,
                    tmp_row_header_list,
                    tmp_col_header_list,
                ) = ([], [], [], [], [], [])

                (
                    cube_op_list,
                    cube_cell_list,
                    cube_num_list,
                    cube_row_header_list,
                    cube_col_header_list,
                ) = table_cube_generator.get_same_row_cube_pattern(
                    OP="CHANGE RATIO",
                    table=table,
                    header_col_num=header_col_num,
                    header_row_num=header_row_num,
                    candidate_rows=candidate_rows,
                    candidate_cols=candidate_cols,
                )
                tmp_op_list += cube_op_list
                tmp_cell_list += cube_cell_list
                tmp_num_list += cube_num_list
                tmp_row_header_list += cube_row_header_list
                tmp_col_header_list += cube_col_header_list

                (
                    cube_op_list,
                    cube_cell_list,
                    cube_num_list,
                    cube_row_header_list,
                    cube_col_header_list,
                ) = table_cube_generator.get_same_col_cube_pattern(
                    OP="CHANGE RATIO",
                    table=table,
                    header_col_num=header_col_num,
                    header_row_num=header_row_num,
                    candidate_rows=candidate_rows,
                    candidate_cols=candidate_cols,
                )
                tmp_op_list += cube_op_list
                tmp_cell_list += cube_cell_list
                tmp_num_list += cube_num_list
                tmp_row_header_list += cube_row_header_list
                tmp_col_header_list += cube_col_header_list

                (
                    cube_op_list,
                    cube_cell_list,
                    cube_num_list,
                    cube_row_header_list,
                    cube_col_header_list,
                ) = table_cube_generator.get_same_row_cube_pattern(
                    OP="CHANGE RATIO",
                    table=table,
                    header_col_num=header_col_num,
                    header_row_num=header_row_num,
                    candidate_rows=candidate_rows,
                    candidate_cols=candidate_cols,
                    all_flag=True,
                )
                tmp_op_list += cube_op_list
                tmp_cell_list += cube_cell_list
                tmp_num_list += cube_num_list
                tmp_row_header_list += cube_row_header_list
                tmp_col_header_list += cube_col_header_list

                change_ratio_list = []
                for item in tmp_num_list:
                    if len(item) == 2:
                        change_ratio_list.append(
                            round((item[1] - item[0]) / item[0] * 100, 2)
                        )
                        tmp_ans_list.append(
                            round((item[1] - item[0]) / item[0] * 100, 2)
                        )

                (
                    tmp_op_list,
                    tmp_cell_list,
                    tmp_num_list,
                    tmp_ans_list,
                    tmp_row_header_list,
                    tmp_col_header_list,
                ) = table_cube_generator.generate_ranked_cube_result(
                    q_text=q_text,
                    op_list=tmp_op_list,
                    cell_list=tmp_cell_list,
                    num_list=tmp_num_list,
                    ans_list=tmp_ans_list,
                    row_header_list=tmp_row_header_list,
                    col_header_list=tmp_col_header_list,
                    top_k=TOP_K if sim_change_ratio != max(sim_list) else TOP_K_1,
                )

                op_list += tmp_op_list
                cell_list += tmp_cell_list
                num_list += tmp_num_list
                ans_list += tmp_ans_list
                row_header_list += tmp_row_header_list
                col_header_list += tmp_col_header_list

            #######################
            #######################
            # Division handle
            if sim_divides > 0:
                num_list = []

                (
                    tmp_op_list,
                    tmp_cell_list,
                    tmp_num_list,
                    tmp_ans_list,
                    tmp_row_header_list,
                    tmp_col_header_list,
                ) = ([], [], [], [], [], [])

                (
                    cube_op_list,
                    cube_cell_list,
                    cube_num_list,
                    cube_row_header_list,
                    cube_col_header_list,
                ) = table_cube_generator.get_same_row_cube_pattern(
                    OP="DIVIDE",
                    table=table,
                    header_col_num=header_col_num,
                    header_row_num=header_row_num,
                    candidate_rows=candidate_rows,
                    candidate_cols=candidate_cols,
                )
                tmp_op_list += cube_op_list
                tmp_cell_list += cube_cell_list
                tmp_num_list += cube_num_list
                tmp_row_header_list += cube_row_header_list
                tmp_col_header_list += cube_col_header_list

                (
                    cube_op_list,
                    cube_cell_list,
                    cube_num_list,
                    cube_row_header_list,
                    cube_col_header_list,
                ) = table_cube_generator.get_same_row_cube_pattern(
                    OP="DIVIDE",
                    table=table,
                    header_col_num=header_col_num,
                    header_row_num=header_row_num,
                    candidate_rows=candidate_rows,
                    candidate_cols=candidate_cols,
                    all_flag=True,
                )
                tmp_op_list += cube_op_list
                tmp_cell_list += cube_cell_list
                tmp_num_list += cube_num_list
                tmp_row_header_list += cube_row_header_list
                tmp_col_header_list += cube_col_header_list

                (
                    cube_op_list,
                    cube_cell_list,
                    cube_num_list,
                    cube_row_header_list,
                    cube_col_header_list,
                ) = table_cube_generator.get_same_col_cube_pattern(
                    OP="DIVIDE",
                    table=table,
                    header_col_num=header_col_num,
                    header_row_num=header_row_num,
                    candidate_rows=candidate_rows,
                    candidate_cols=candidate_cols,
                )
                tmp_op_list += cube_op_list
                tmp_cell_list += cube_cell_list
                tmp_num_list += cube_num_list
                tmp_row_header_list += cube_row_header_list
                tmp_col_header_list += cube_col_header_list

                (
                    cube_op_list,
                    cube_cell_list,
                    cube_num_list,
                    cube_row_header_list,
                    cube_col_header_list,
                ) = table_cube_generator.get_same_col_cube_pattern(
                    OP="DIVIDE",
                    table=table,
                    header_col_num=header_col_num,
                    header_row_num=header_row_num,
                    candidate_rows=candidate_rows,
                    candidate_cols=candidate_cols,
                    all_flag=True,
                )
                tmp_op_list += cube_op_list
                tmp_cell_list += cube_cell_list
                tmp_num_list += cube_num_list
                tmp_row_header_list += cube_row_header_list
                tmp_col_header_list += cube_col_header_list

                divide_list = []
                for item in tmp_num_list:
                    if len(item) == 2:
                        if "proportion" or "ratio" in q_text:
                            divide_list.append(round(item[0] / (item[1] + 1e-8), 2))
                            tmp_ans_list.append(round(item[0] / (item[1] + 1e-8), 2))
                        else:
                            divide_list.append(
                                round(item[0] / (item[1] + 1e-8) * 100, 2)
                            )
                            tmp_ans_list.append(
                                round(item[0] / (item[1] + 1e-8) * 100, 2)
                            )
                (
                    tmp_op_list,
                    tmp_cell_list,
                    tmp_num_list,
                    tmp_ans_list,
                    tmp_row_header_list,
                    tmp_col_header_list,
                ) = table_cube_generator.generate_ranked_cube_result(
                    q_text=q_text,
                    op_list=tmp_op_list,
                    cell_list=tmp_cell_list,
                    num_list=tmp_num_list,
                    ans_list=tmp_ans_list,
                    row_header_list=tmp_row_header_list,
                    col_header_list=tmp_col_header_list,
                    top_k=TOP_K if sim_change_ratio != max(sim_list) else TOP_K_1,
                )

                op_list += tmp_op_list
                cell_list += tmp_cell_list
                num_list += tmp_num_list
                ans_list += tmp_ans_list
                row_header_list += tmp_row_header_list
                col_header_list += tmp_col_header_list

            assert (
                len(op_list) == len(cell_list)
                and len(cell_list) == len(ans_list)
                and len(ans_list) == len(row_header_list)
                and len(row_header_list) == len(col_header_list)
            )

            cube_results = []
            for op_, cells_, nums_, row_header_, col_header_ in zip(
                op_list, cell_list, ans_list, row_header_list, col_header_list
            ):
                cube_results.append(
                    {
                        "operation": op_,
                        "cell_coordinates": cells_,
                        "pseudo_answer": nums_,
                        "row_header": row_header_,
                        "col_header": col_header_,
                    }
                )

            data_item_new["questions"][iquestion].update(
                {
                    "cube": cube_results,
                }
            )

        data_out.append(data_item_new)

    json.dump(data_out, f_out_path, indent=4, cls=NpEncoder)


if __name__ == "__main__":

    main()
