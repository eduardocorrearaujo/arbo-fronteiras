import pandas as pd


def fill_missing_dates(data, format="%Y-%m-%d"):
    """Fills missing dates from a pandas dataframe. 

    Parameters
    ----------
    data : pandas.DataFrame
        Data whose index is a datetime column.

    Returns
    -------
    data : pandas.DataFrame
        Data reindexed with filled missing data. 

    Notes
    -----
    The dataframe column to be filled should be in data.index.
    """
    try:
        data.index = pd.to_datetime(data.index, format=format)
    except ParserError:
        raise

    date_min_max = pd.date_range(
        data.index.min(),
        data.index.max()
    )
    data = data.reindex(date_min_max, fill_value=0)

    return data
