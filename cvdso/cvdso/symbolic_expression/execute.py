from scibench import cyfunc

import array

def cython_execute(traversal, X):
    """
    Execute cython function using given traversal over input X.

    Parameters
    ----------

    traversal : list
        A list of nodes representing the traversal over a Program.
    X : np.array
        The input values to execute the traversal over.

    Returns
    -------

    result : float
        The result of executing the traversal.
    """
    is_input_var = array.array('i', [t.input_var is not None for t in traversal])
    return cyfunc.execute(X, len(traversal), traversal, is_input_var)


