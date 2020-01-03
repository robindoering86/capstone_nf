def check_for_missing_vals(df):
    cnt = 0
    if df.columns.nlevels == 1:
        ls = list(df.isna().sum())
        if sum(ls) != 0:
            print(ls)
            cnt += 1
    else:
        for x in list(set(df.columns.get_level_values(0))):
            ls = list(df[x].isna().sum())
            if sum(ls) != 0:
                print(x)
                print(ls)
                cnt += 1
    if cnt == 0:
        print('No missing values found in dataframe')