# Imports

def calcIndicators(data, symbol, lookback_windows, ti_list):
    '''
    Computes technical indicators for a given data set and 
    lookback windows. 
    :param data dataframe with OHLC data. Expects MultiIndex
    :param lookback_windows: List of time lookback time windows
    :param: ti_list List of technical indicators to compute
    '''
    from atomm.Indicators import MomentumIndicators
    import pandas as pd
    import numpy as np
    
    if not isinstance(data, pd.DataFrame):
        raise ValueError('data must be a Dataframe.')
    mi = MomentumIndicators(data[symbol])
    df = pd.DataFrame(index=data.index)
    ti_dict = {
        'sma': MomentumIndicators(data[symbol]).calcSMA,
        'macd': MomentumIndicators(data[symbol]).calcMACD,
           'rsi': MomentumIndicators(data[symbol]).calcRSI,
           'stoc': MomentumIndicators(data[symbol]).calcSTOC,
           'roc': MomentumIndicators(data[symbol]).calcROC,
           'bb': MomentumIndicators(data[symbol]).calcBB,
           'ema': MomentumIndicators(data[symbol]).calcEMA,
           'atr': MomentumIndicators(data[symbol]).calcATR,
           'adx': MomentumIndicators(data[symbol]).calcADX,
           'cci': MomentumIndicators(data[symbol]).calcCCI,
           'williamsr': MomentumIndicators(data[symbol]).calcWR,
            }
    for n in lookback_windows:
        for ti in ti_list:
            if ti == 'stocd':
                df[f'{ti}_{n}'] = mi.calcEMA(3, df[f'stoc_{n}'])
            elif ti == 'log_ret':
                df[f'{ti}_{n}'] = np.log(data[symbol]['Close']).diff(periods=n)
            elif ti in ['autocorr_1', 'autocorr_3', 'autocorr_5']:
                df[f'{ti}_{n}'] = df[f'log_ret_{n}'].rolling(
                    window=n,
                    min_periods=n,
                    center=False
                ).apply(lambda x: x.autocorr(lag=int(ti[-1])), raw=False)
            elif ti == 'vol':
                df[f'vol_{n}'] = df[f'log_ret_{n}'].rolling(
                    window=n,
                    min_periods=n,
                    center=False
                ).std()
            elif ti == 'macd':
                df[f'macd_{n}'] = ti_dict.get(ti)(n, n*2)
            elif ti == 'bb':
                df[f'bbu_{n}'], df[f'bbu_{n}'], _ = ti_dict.get(ti)(n, 2)
            elif ti == 'arima':
                arima = np.empty(data.shape[0],)
                for t in range(n, data.shape[0]):
                    pred = ARIMAtune(data[symbol]['Close'][t-n:t+1], lookahead=[1], d_lookback=n)
                    #score = eval(pred['preds'].iloc[pred['mse'].fillna(np.inf).idxmin()])[-1]
                    arima[t] = pred
                df[f'arima_{n}'] = arima
                df[f'arima_{n}'] = np.log(df[f'arima_{n}']).diff().fillna(0)
            else:
                df[f'{ti}_{n}'] = ti_dict.get(ti)(n)

            # Shynkevich et al 2017 also has SMA, but not Bollinger Bands, MACD
    return df
 
def ARIMAeval(data, lookahead, d_lookback, order):
    from statsmodels.tsa.arima_model import ARIMA, ARMA
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import pandas as pd
    preds, trues = [], []
    lookahead = [1]
    try:
        model = ARIMA(data[:d_lookback].values, order=order)
        model_fit = model.fit(disp=0,)  
        for step in lookahead:
            try:
                prediction = model_fit.forecast(steps=step)
                pred = prediction[0][-1]
                true = data[d_lookback+step]
                preds.append(pred)
                trues.append(true)
            except:
                preds.append(0)
                trues.append(1000)
                continue
        error = mean_squared_error(trues, preds)
    except: 
        trues = [1000 for i in range(len(lookahead))]
        preds = [0 for i in range(len(lookahead))]
        error = 1E15   
    return preds, error
    

def ARIMAtune(data, lookahead = [1, 3, 5, 7, 10, 15, 20, 25, 30], d_lookback = 30, p = 4, d = 2, q = 4):
    from statsmodels.tsa.arima_model import ARIMA, ARMA
    from sklearn.metrics import mean_squared_error
    import numpy as np
    import pandas as pd
    np.append(np.array([1,2]), 4).argmin()
    length = p*d*q
    preds, ps, ds, qs, errors = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    for p in range(p):
        for d in range(d):
            for q in range(q):
                order = (p, d, q)
                ps = np.append(ps, p)
                ds = np.append(ds, d)
                qs = np.append(qs, q)
                pred, score = ARIMAeval(data, lookahead=lookahead, d_lookback=d_lookback, order=order)
                errors = np.append(errors, score)
                preds = np.append(preds, pred)
                
    #df = pd.DataFrame([ps, ds, qs, preds, errors]).T
    #df.columns = ['p', 'd', 'q', 'preds', 'mse']
    
    # returns best prediction
    pred = preds[errors.argmin()]
    return pred