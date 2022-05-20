from typing import List


def count_marginal_objects(
    lines: List[List],
    marginal_classes: List[str],
):
    """
    Returns the count of lines for which the class_id falls into
    the marginal_classes group.

    """
    count_marginal = 0
    for line in lines:
        class_id = str(line[0]).strip()
        if class_id in marginal_classes or int(class_id) in marginal_classes:
            count_marginal += 1
    return count_marginal


def insufficient_expectation(
    new_lines: List[List],
    marginal_classes: List[str],
    min_count_marginal: int,
) -> bool:
    """
    If count_objects < min_count_marginal we could have a problem with per image precision
    if ALL of those objects pertain to marginal classes.

    We can improve our odds of getting at least one detection correct if we apply
    a minimum count.

    NOTE::
        Another approach would be to calculate expected precision
        = Sum_i_n(precision_i * count_i) where i is the index for the class
        and n is the complete set of classes.

    """
    num_objects = len(new_lines)
    num_marginal = count_marginal_objects(
        lines=new_lines,
        marginal_classes=marginal_classes,
    )
    num_precise_objects = num_objects - num_marginal
    if num_precise_objects == 0 and num_marginal < min_count_marginal:
        return True
    else:
        return False
