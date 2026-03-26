import pandas as pd

def load_and_prepare_dataset(path):
    """
    Load dataset from CSV and transform pipe-separated fields into lists.
    Parameters
    ----------
    path : str
        File path to the CSV dataset.
    Returns
    -------
    pandas.DataFrame
        DataFrame with 'genres' and 'keywords' columns converted to lists.
    """
    # Load dataset
    df = pd.read_csv(path)
    def split_pipe_to_list(text):
        """
        Convert pipe-separated string into a list of cleaned strings.
        Parameters
        ----------
        text : str or NaN
        Returns
        -------
        list
            List of cleaned string elements.
        """
        if pd.isna(text) or text == "":
            return []
        return [item.strip() for item in text.split("|") if item.strip()]
    # Transform columns if present
    if "genres" in df.columns:
        df["genres"] = df["genres"].apply(split_pipe_to_list)
    if "keywords" in df.columns:
        df["keywords"] = df["keywords"].apply(split_pipe_to_list)
    return df