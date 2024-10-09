import re


def validate_position_string(s):
    """
    validate intervention_positions argument in the form of "l2", "f3", or "f3+l2"
    """
    single_pattern = re.compile(r'^[fl]\d+$')
    combined_pattern = re.compile(r'^(f\d+)\+(l\d+)$')

    if single_pattern.match(s) or combined_pattern.match(s):
        return True
    else:
        raise ValueError("The string must be in the format e.g. 'f2', 'l3' or 'f2+l4'")


def parse_positions(positions: str):
    validate_position_string(positions)
    # parse position
    first_n, last_n = 0, 0
    if "+" in positions:
        first_n = int(positions.split("+")[0].strip("f"))
        last_n = int(positions.split("+")[1].strip("l"))
    else:
        if "f" in positions:
            first_n = int(positions.strip("f"))
        elif "l" in positions:
            last_n = int(positions.strip("l"))
    return first_n, last_n


def get_intervention_locations(**kwargs):
    """
    This function generates the intervention locations.

    For your customized dataset, you want to create your own function.
    """
    # parse kwargs
    share_weights = kwargs["share_weights"] if "share_weights" in kwargs else False
    last_position = kwargs["last_position"]
    if "positions" in kwargs:
        _first_n, _last_n = parse_positions(kwargs["positions"])
    else:
        _first_n, _last_n = kwargs["first_n"], kwargs["last_n"]
    num_interventions = kwargs["num_interventions"]
    pad_mode = kwargs["pad_mode"] if "pad_mode" in kwargs else "first"

    first_n = min(last_position // 2, _first_n)
    last_n = min(last_position // 2, _last_n)

    pad_amount = (_first_n - first_n) + (_last_n - last_n)
    pad_position = -1 if pad_mode == "first" else last_position
    if share_weights or (first_n == 0 or last_n == 0):
        position_list = [i for i in range(first_n)] + \
            [i for i in range(last_position - last_n, last_position)] + \
            [pad_position for _ in range(pad_amount)]
        intervention_locations = [position_list]*num_interventions
    else:
        left_pad_amount = (_first_n - first_n)
        right_pad_amount = (_last_n - last_n)
        left_intervention_locations = [i for i in range(first_n)] + [pad_position for _ in range(left_pad_amount)]
        right_intervention_locations = [i for i in range(last_position - last_n, last_position)] + \
            [pad_position for _ in range(right_pad_amount)]
        # after padding, there could be still length diff, we need to do another check
        left_len = len(left_intervention_locations)
        right_len = len(right_intervention_locations)
        if left_len > right_len:
            right_intervention_locations += [pad_position for _ in range(left_len-right_len)]
        else:
            left_intervention_locations += [pad_position for _ in range(right_len-left_len)]
        intervention_locations = [left_intervention_locations]*(num_interventions//2) + \
            [right_intervention_locations]*(num_interventions//2)
    
    return intervention_locations


def replace_pad_in_2d_list_intervention_locations(intervention_locations, replace_ele_type="left", pad_mark=-1):

    for sublist in intervention_locations:
        replace_ele = None
        if replace_ele_type == "left":
            for p in sublist:
                if p != pad_mark:
                    replace_ele = p
                    break    
        elif replace_ele_type == "right":
            for p in sublist[::-1]:
                if p != pad_mark:
                    replace_ele = p
                    break
        else:
            raise ValueError(f"The type of replace element \"{replace_ele_type}\" is unsupported!")
        assert replace_ele is not None, "There is no valid position values in the list."

        sublist[:] = [replace_ele if x == pad_mark else x for x in sublist]
