import scipy.stats
import pandas as pd


class Data_hlpr:
    """Data Helper is a tool with various static methods help you process and understand your data.
    """

    @staticmethod
    def pd_expand_dict(df: pd.core.frame.DataFrame, dict_col: str) -> pd.core.frame.DataFrame:
        """
        Expands a dictionary column into 1 to many dataframe columns. Typically, used when
        applying classifiers onto datasets {"Class": "Apple", "Prob":.44}

        Parameters
        ----------
        df: pd.core.frame.DataFrame
             Dataframe
        dict_col: str
            The name of the column containing a dictionary
        Returns
        -------
        result: pd.core.frame.DataFrame
            A dataframe containing keys as the column names and their corresponding values as row observations.
        """

        df[dict_col] = df[dict_col].astype(str)
        df[dict_col] = df[dict_col].apply(lambda x: dict(eval(x)))
        df2 = df[dict_col].apply(pd.Series)
        result = pd.concat([df, df2], axis=1).drop(dict_col, axis=1)

        return(result)

    @staticmethod
    def dummycode(df: pd.core.frame.DataFrame, dummycode: str) -> pd.core.frame.DataFrame:
        """
        Converts categorical variables of pandas dataframe into their own binary columns.

        Parameters
        ----------
        df:  pd.core.frame.DataFrame
             dataframe
        dummycode: str
            The categorical column that will be transposed into a wide binary format.
        Returns
        -------
        dummies: list
            A list of lists.
        """
        dummies = pd.get_dummies(df[dummycode])
        dummies = pd.concat([df, dummies], axis=1)
        dummies = dummies.drop([dummycode], axis=1)

        return(dummies)

    @staticmethod
    def dataframestats(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        """
        Calculates summary stats on a dataframe.

        Parameters
        ----------
        df:  pd.core.frame.DataFrame
             dataframe
        Returns
        -------
        types: d.core.frame.DataFrame
            A dataframe containg summary stats
        """
        types = pd.DataFrame(df.dtypes)
        types = types.reset_index()

        types.columns = ['VarName', 'VarType']

        types = types.reindex(
            columns=[
                *types.columns.tolist(),
                'CompleteRecords',
                'MissingRecords',
                'Average',
                'Median',
                'StandardDeviation',
                'Min',
                'Max',
                'UniqueObs',
                'Skewness'],
            fill_value='')

        for i in range(0, len(types)):
            types.loc[i, 'MissingRecords'] = sum(pd.isnull(df[types['VarName'][i]]).values.ravel())
            types.loc[i, 'CompleteRecords'] = sum(df[types['VarName'][i]].notnull().values.ravel())
            if str(df[types['VarName'][i]].dtype) != 'object':
                types.loc[i, 'Average'] = df[types['VarName'][i]].mean()
                types.loc[i, 'Median'] = df[types['VarName'][i]].median()
                types.loc[i, 'Min'] = min(df[types['VarName'][i]])
                types.loc[i, 'Max'] = max(df[types['VarName'][i]])
                types.loc[i, 'StandardDeviation'] = df[types['VarName'][i]].std()
                types.loc[i, 'Skewness'] = scipy.stats.skew(df[types['VarName'][i]], axis=0, bias=True)
                types.loc[i, 'UniqueObs'] = len(pd.Series(df[types['VarName'][i]].ravel()).unique())
        types = types.fillna('')

        return types

    @staticmethod
    def group_df(df: pd.core.frame.DataFrame, cols: list) -> pd.core.frame.DataFrame:
        """
        Counts values within a column.

        Parameters
        ----------
        df:  pd.core.frame.DataFrame
             dataframe
        cols: list
            The columns you want to aggregate by
        Returns
        -------
        output: pd.core.frame.DataFrame
            A dataframe with the counts by the input columns.
        """

        end_pos = len(cols)
        output = df.groupby(cols).count().reset_index().iloc[:, 0:end_pos + 1]
        output.columns.values[end_pos] = "Count"

        return output
