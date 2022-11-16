import inflect
from fuzzywuzzy import fuzz

from generator.cube_generator import CubeGenerator
from utils.fuzz_retrieve import get_order_by_tf_idf
from utils.tatqa.extract_num_utils import *

class TaTQAGenerator(CubeGenerator):
    def __init__(self) -> None:
        self.operand_2_list = {"DIFF", "DIVIDE", "CHANGE RATIO"}
        self.inflect = inflect.engine()

    def _clean_cell(self, text: str):
        EXCLUDE_IN_NUM = "'\"\\$€£¥%(),[]"
        return "".join([ch for ch in str(text) if ch not in EXCLUDE_IN_NUM]).lower()

    def get_candidate_row_col_from_table(
        self, q_text, table, header_row_num, header_col_num, threshold=50
    ):
        candidate_rows = []
        candidate_rows_header_col = []
        candidate_cols = []
        candidate_cols_header_row = []

        for irow, row in enumerate(table):
            for icol, cell in enumerate(row):
                if irow < header_row_num:
                    cell = self._clean_cell(cell)
                    _, sim_cell = get_order_by_tf_idf(q_text, [cell])
                    fuzz_score = fuzz.partial_ratio(q_text, cell)
                    sim_cell = sim_cell[0][1]
                    if sim_cell > 0 and fuzz_score >= threshold:
                        candidate_cols.append(icol)
                        candidate_rows_header_col.append(irow)

        for irow, row in enumerate(table):
            for icol, cell in enumerate(row):
                if icol < header_col_num:
                    _, sim_cell = get_order_by_tf_idf(q_text, [cell])
                    fuzz_score = fuzz.partial_ratio(q_text, cell)
                    sim_cell = sim_cell[0][1]
                    if sim_cell > 0 and fuzz_score >= threshold:
                        candidate_rows.append(irow)
                        candidate_cols_header_row.append(icol)

        candidate_rows = sorted(list(set(candidate_rows)))
        candidate_cols = sorted(list(set(candidate_cols)))
        candidate_rows_header_col = sorted(list(set(candidate_rows_header_col)))
        candidate_cols_header_row = sorted(list(set(candidate_cols_header_row)))

        return (
            candidate_rows,
            candidate_cols,
            candidate_rows_header_col,
            candidate_cols_header_row,
        )

    def get_row_header_name(self, table, irow, num_header_columns):
        ret = []
        for i in range(num_header_columns):
            ret.append(table[irow][i])
        return ret

    def get_col_header_name(self, table, icol, num_header_rows):
        ret = []
        for i in range(num_header_rows):
            ret.append(table[i][icol])
        return ret

    def get_same_row_cube_pattern(
        self,
        OP,
        table,
        header_col_num,
        header_row_num,
        candidate_rows,
        candidate_cols,
        all_flag=False,
    ):

        op_list = []
        cell_list = []
        num_list = []
        row_header_list = []
        col_header_list = []
        for irow in candidate_rows:
            cur_list = []
            cur_row_header = [
                self.get_row_header_name(table, irow, header_col_num)
            ]  # return a list
            cur_col_header = []
            cur_cells = []

            # "all column" pattern
            if all_flag:
                candidate_cols = [icol for icol in range(len(table[0]))]
            for icol in candidate_cols:
                cell = table[irow][icol]
                if to_number(cell) and is_pure_number(cell):
                    cur_list.append(abs(to_number(cell)))
                    cur_col_header += [
                        self.get_col_header_name(table, icol, header_row_num)
                    ]
                    cur_cells.append([irow, icol])

            if OP not in self.operand_2_list and len(cur_list) != 0:
                op_list.append(OP)
                cell_list.append(cur_cells)
                num_list.append(cur_list)
                row_header_list.append(cur_row_header)
                if all_flag:
                    col_header_list.append([["ALL"]])
                else:
                    col_header_list.append(cur_col_header)
            elif OP in self.operand_2_list:
                if len(cur_list) == 2 and not all_flag:  # 2-operand naive
                    op_list.append(OP)
                    cell_list.append(cur_cells)
                    num_list.append(cur_list)
                    row_header_list.append(cur_row_header)
                    col_header_list.append(cur_col_header)

                    op_list.append(OP)
                    cell_list.append(cur_cells[::-1])
                    num_list.append(cur_list[::-1])
                    row_header_list.append(cur_row_header)
                    col_header_list.append(cur_col_header[::-1])
                elif (
                    len(cur_list) >= 2 and all_flag
                ):  # 2-operand need to brute force search
                    (
                        op_list_,
                        cell_list_,
                        num_list_,
                        row_header_list_,
                        col_header_list_,
                    ) = ([], [], [], [], [])
                    for idx_1 in range(len(cur_list)):
                        for idx_2 in range(idx_1 + 1, len(cur_list)):
                            op_list_.append(OP)
                            cell_list_.append([cur_cells[idx_1], cur_cells[idx_2]])
                            num_list_.append([cur_list[idx_1], cur_list[idx_2]])
                            row_header_list_.append(cur_row_header)
                            col_header_list_.append(
                                [cur_col_header[idx_1], cur_col_header[idx_2]]
                            )

                            # reverse order
                            op_list_.append(OP)
                            cell_list_.append([cur_cells[idx_2], cur_cells[idx_1]])
                            num_list_.append([cur_list[idx_2], cur_list[idx_1]])
                            row_header_list_.append(cur_row_header)
                            col_header_list_.append(
                                [cur_col_header[idx_2], cur_col_header[idx_1]]
                            )

                    op_list += op_list_
                    cell_list += cell_list_
                    num_list += num_list_
                    row_header_list += row_header_list_
                    col_header_list += col_header_list_

        return op_list, cell_list, num_list, row_header_list, col_header_list

    def get_same_col_cube_pattern(
        self,
        OP,
        table,
        header_col_num,
        header_row_num,
        candidate_rows,
        candidate_cols,
        all_flag=False,
    ):
        op_list, cell_list, num_list, row_header_list, col_header_list = (
            [],
            [],
            [],
            [],
            [],
        )
        for icol in candidate_cols:
            cur_list = []
            cur_col_header = [self.get_col_header_name(table, icol, header_row_num)]
            cur_row_header = []
            cur_cells = []
            # all rows in candidate cols, for special patterns
            if all_flag:
                candidate_rows = [irow for irow in range(len(table))]
            for irow in candidate_rows:
                cell = table[irow][icol]

                if to_number(cell) and is_pure_number(cell):
                    cur_list.append(abs(to_number(cell)))
                    cur_row_header += [
                        self.get_row_header_name(table, irow, header_col_num)
                    ]
                    cur_cells.append([irow, icol])

            if OP not in self.operand_2_list and len(cur_list) != 0:
                op_list.append(OP)
                cell_list.append(cur_cells)
                num_list.append(cur_list)
                if all_flag:
                    row_header_list.append([["ALL"]])
                else:
                    row_header_list.append(cur_row_header)
                col_header_list.append(cur_col_header)
            elif OP in self.operand_2_list:
                if len(cur_list) == 2 and not all_flag:  # 2-operand naive
                    op_list.append(OP)
                    cell_list.append(cur_cells)
                    num_list.append(cur_list)
                    row_header_list.append(cur_row_header)
                    col_header_list.append(cur_col_header)

                    op_list.append(OP)
                    cell_list.append(cur_cells[::-1])
                    num_list.append(cur_list[::-1])
                    row_header_list.append(cur_row_header[::-1])
                    col_header_list.append(cur_col_header)
                elif (
                    len(cur_list) >= 2 and all_flag
                ):  # 2-operand need to brute force search
                    (
                        op_list_,
                        cell_list_,
                        num_list_,
                        row_header_list_,
                        col_header_list_,
                    ) = ([], [], [], [], [])
                    for idx_1 in range(len(cur_list)):
                        for idx_2 in range(idx_1 + 1, len(cur_list)):
                            op_list_.append(OP)
                            cell_list_.append([cur_cells[idx_1], cur_cells[idx_2]])
                            num_list_.append([cur_list[idx_1], cur_list[idx_2]])
                            row_header_list_.append(
                                [cur_row_header[idx_1], cur_row_header[idx_2]]
                            )
                            col_header_list_.append(cur_col_header)

                            # reverse order
                            op_list_.append(OP)
                            cell_list_.append([cur_cells[idx_2], cur_cells[idx_1]])
                            num_list_.append([cur_list[idx_2], cur_list[idx_1]])
                            row_header_list_.append(
                                [cur_row_header[idx_2], cur_row_header[idx_1]]
                            )
                            col_header_list_.append(cur_col_header)

                    op_list += op_list_
                    cell_list += cell_list_
                    num_list += num_list_
                    row_header_list += row_header_list_
                    col_header_list += col_header_list_

        return op_list, cell_list, num_list, row_header_list, col_header_list

    def generate_ranked_cube_result(
        self,
        q_text,
        op_list,
        cell_list,
        num_list,
        ans_list,
        row_header_list,
        col_header_list,
        top_k=None,
    ):
        row_sequences = []
        for (
            headers
        ) in row_header_list:  # sample wise headers like [['xxx','xxx'], ['xxx, 'xxx]]
            row_sequence = []
            for row_headers in headers:  # cell wise row headers like ['xxx', 'xxx']
                sequence = " ".join(row_headers)  # header wise
                row_sequence.append(sequence)
            row_sequence = " ".join(row_sequence)
            row_sequences.append(row_sequence)

        col_sequences = []
        for (
            headers
        ) in col_header_list:  # sample wise headers like [['xxx','xxx'], ['xxx, 'xxx]]
            col_sequence = []
            for col_headers in headers:  # cell wise row headers like ['xxx', 'xxx']
                sequence = " ".join(col_headers)  # header wise
                col_sequence.append(sequence)
            col_sequence = " ".join(col_sequence)
            col_sequences.append(col_sequence)
        cube_sequence = [
            " ".join([op, row_headers, col_headers])
            for op, row_headers, col_headers in zip(
                op_list, row_sequences, col_sequences
            )
        ]

        order_idx, sorted_similarities = get_order_by_tf_idf(q_text, cube_sequence)
        if top_k is not None:
            order_idx = order_idx[: min(top_k, len(order_idx))]

        op_list = [op_list[idx] for idx in order_idx]
        cell_list = [cell_list[idx] for idx in order_idx]
        num_list = [num_list[idx] for idx in order_idx]
        ans_list = [ans_list[idx] for idx in order_idx]
        row_header_list = [row_header_list[idx] for idx in order_idx]
        col_header_list = [col_header_list[idx] for idx in order_idx]

        return op_list, cell_list, num_list, ans_list, row_header_list, col_header_list
