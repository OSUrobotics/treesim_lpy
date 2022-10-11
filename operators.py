from stochastic_tree import BasicWood

def tie_at(lstring, idx, guide, depth=1):
    """
    Given an L-String, the index of the branch to be guided, and a curve to be followed,
    update the existing guide.
    """

    # Check if there is a rotation/existing guide following the current object
    # If so, remove it
    for _ in range(depth):
        if lstring[idx + 1].name in ['&', '/', 'SetGuide']:
            del lstring[idx + 1]

    lstring.insertAt(idx + 1, guide)



def tie_all(lstring):
    for idx, lsys_elem in enumerate(lstring):
        try:
            wood = lsys_elem[0].type
        except (TypeError, AttributeError, IndexError):
            continue

        if not isinstance(wood, BasicWood) or not wood.can_tie:
            continue

        # If an element has not had its tie updated, we continue to let it grow out
        if not wood.tie_updated or not wood.guide_points:
            continue

        guide = wood.produce_tie_guide()
        wood.tie_updated = False

        depth = 1
        if not wood.has_tied:
            depth = 2
            wood.has_tied = True

        tie_at(lstring, idx, guide, depth=depth)

        return True
    return False
