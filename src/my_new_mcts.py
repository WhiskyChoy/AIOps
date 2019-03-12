TOTAL_CHOICES = []
USED_CHOICES = set()
CURRENT_MAX_QS = 0

MAX_ROUND_NUMBER = 10
# 0 < PT <= 1
PT = 0.9


class Node:
    def __init__(self):
        # For the first root node, the index is 0 and the game should start from 1
        self.parent = None
        self.children = []
        self.visit_times = 0
        # TODO 这里的Q是max算法
        self.quality_value = 0.0
        self.tested_set = {}

        def node_all_expanded(self):
            return


def all_expanded():
    return len(USED_CHOICES) == 2 ** len(TOTAL_CHOICES) - 1


def get_new_set(current_set):
    if all_expanded():
        return None
    for choice in TOTAL_CHOICES:
        result = current_set + choice
        if result not in USED_CHOICES:
            USED_CHOICES.add(result)
            return result


def selection(node):
    return node


def expansion(node):
    return node


def evaluation(node):
    return False


def backup(node):
    return node


def monte_carlo_tree_search(node):
    for i in range(MAX_ROUND_NUMBER):
        selected_node = selection(node)

        expanded_node = expansion(selected_node)

        best_set_found = evaluation(expanded_node)

        if best_set_found:
            return {'best_set_found': True, 'target_node': expanded_node}

        backup(expanded_node)
    # TODO 取max qs
    return {'best_set_found': False, 'target_node': None}


def do_monte_carlo_search(available_choices, max_round_num=None, pt_num=None):
    global TOTAL_CHOICES
    global MAX_ROUND_NUMBER
    global PT
    global CURRENT_MAX_QS
    global USED_CHOICES
    TOTAL_CHOICES = available_choices
    CURRENT_MAX_QS = 0
    USED_CHOICES = set()

    if max_round_num is not None:
        MAX_ROUND_NUMBER = max_round_num
    if pt_num is not None:
        PT = pt_num

    init_node = Node()
    selected_node = monte_carlo_tree_search(init_node)
    print(selected_node)
