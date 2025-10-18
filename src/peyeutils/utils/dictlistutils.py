def list_of_dicts_to_dict_of_lists( mylistdict ):
    """

    Parameters
    ----------
    mylistdict :
        

    Returns
    -------

    """
    listdict = dict();
    for key in mylistdict[0]:
        listdict[key] = [ i[key] for i in mylistdict ];
        pass;
    return listdict;
