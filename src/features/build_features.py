# 
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
            else:
                df[f'{ti}_{n}'] = ti_dict.get(ti)(n)

            # Shynkevich et al 2017 also has SMA, but not Bollinger Bands, MACD
    return df

def arimaTrend(data, d_lookback=3, d_lookahead=1):
    """
    :param data: Pandas DataFrame or Series
    :param d_lookback
    :param d_lookahead
    :return Price trend for d_lookahead
    """
    from statsmodels.tsa.arima_model import ARIMA
    X = data.values

    predictions = []
    for t in range(data.shape[0]):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
 
