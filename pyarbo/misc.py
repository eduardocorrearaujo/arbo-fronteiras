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


def return_dengue_cases(data):
    """Return dengue cases, classified in notified, probable, and lab confirmed.

    Parameters
    ----------
    data : pandas.DataFrame
        Data .

    Returns
    -------


    Notes
    -----
    Expected columns: "dt_sin_pri", "classi_fin", "criterio"
    """
    data = data.assign(tipo = "notified")

    classifin_not_5 = (data["classi_fin"] != 5)
    criterio_is_1 = (data["criterio"] == 1)

    data.loc[:, (classifin_not_5, "tipo")] = "probable"

    dengue_mask = classifin_not_5 & criterio_is_1
    data[dengue_mask]["tipo"] = "lab_confirmed"

    return data
