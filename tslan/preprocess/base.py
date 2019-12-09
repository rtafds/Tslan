

def add_datepart(df, fldname, drop=True, time=False):
    """
    Separate the date and add the year, month, day, number of days from the contract month, etc. to the column.
    df: data frame
    fldname: Name of the column containing Date
    I don't need the number of days until the contract month
    No dependency
    """
    # Convert to pandas.timestamp
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64
    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    
    # Extract everything you can withdraw from pandas datetime by default
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',]
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())    
    #df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    
    if drop: df.drop(fldname, axis=1, inplace=True)
    
    
def pd_insert(original_df, insert_df, insert_index, axis=0):
    """Insert rows or columns with pandas"""
    new_df = original_df.copy()
    new_df.reset_index(drop=True, inplace=True)  # When concat, there is a bug if the index is not aligned, so align it. 
    insert_df.reset_index(drop=True, inplace=True)
    if axis==0:
        for i,index in enumerate(insert_index):
            previous_df = new_df.iloc[:insert_index, :]
            behind_df = new_df.iloc[insert_index:,:]
            insert_row = insert_df.iloc[[i], :]
            new_df = pd.concat([previous_df, insert_row, behind_df], axis=0)
    elif axis==1:
        for i,index in enumerate(insert_index):
            previous_df = new_df.iloc[:, :index]
            behind_df = new_df.iloc[:,index:]
            insert_row = insert_df.iloc[:,[i]]
            new_df = pd.concat([previous_df, insert_row, behind_df], axis=1)
    else:
        raise ValueError("axis is 0 or 1")
    return new_df

def seasonal_predict(seasonal_data, cycle, predict_n_row):
    """Return the data from the latest data to be repeated.
    Premise that train_data and the predicted data are continuous
    seasonal_data: Extracted only from the train data season
    cycle: Seasonal correlation cycle. freq
    predict_n_row: How many days do you want?
    No dependency
    """
    
    seasonal_data = pd.Series(seasonal_data)
    cycle_data = seasonal_data.iloc[:cycle]
    cycle2_data = seasonal_data.iloc[cycle:cycle*2]
    if not all([a==b for a,b in zip(cycle_data, cycle2_data)]):  # When the value is different in the cycle
        raise Exception("""There are not cycle data or cycle is incorrect""")
    
    rest = seasonal_data.shape[0]%cycle  # Where the cycle ends
    predict=cycle_data.iloc[rest:]  # From the middle of the cycle
    for i in range(predict_n_row//cycle+1):
        predict=pd.concat([predict, cycle_data],axis=0)
    return np.array(predict[:predict_n_row])