import numpy as np

def moving_average(x, n, type='simple'):
    """
    compute an n period moving average.

    type is 'simple' | 'exponential'

    """
    x = np.asarray(x)
    if type == 'simple':
        weights = np.ones(n)
    else:
        weights = np.exp(np.linspace(-1., 0., n))

    weights /= weights.sum()

    a = np.convolve(x, weights, mode='full')[:len(x)]
    a[:n] = a[n]
    return a


def relative_strength(prices, n=14):
    """
    compute the n period relative strength indicator
    http://stockcharts.com/school/doku.php?id=chart_school:glossary_r#relativestrengthindex
    http://www.investopedia.com/terms/r/rsi.asp
    """

    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed >= 0].sum()/n
    down = -seed[seed < 0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1. + rs)

    for i in range(n, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n - 1) + upval)/n
        down = (down*(n - 1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)

    return rsi


def moving_average_convergence(x, nslow=26, nfast=12):
    """
    compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
    return value is emaslow, emafast, macd which are len(x) arrays
    """
    emaslow = moving_average(x, nslow, type='exponential')
    emafast = moving_average(x, nfast, type='exponential')
    return emaslow, emafast, emafast - emaslow

def ref_macd_crossover(px,nFast = 12, nSlow = 26, nSig = 9):
	""" MACD Signal Line & Centerline Crossovers
		https://www.linkedin.com/pulse/python-tutorial-macd-signal-line-centerline-andrew-hamlet
		reference only not for daily use
	"""
	df=px.to_frame(name="price")
	(df['26slow'],df['12fast'],df['macdNew'])=moving_average_convergence(df["price"])
	df['rsi']=relative_strength(df["price"])
	df['26 ema'] = pd.ewma(df["price"], span=nSlow) 
	df['12 ema'] = pd.ewma(df["price"], span=nFast) 
	df['MACD'] = (df['12 ema'] - df['26 ema'])
	df['Signal Line'] = pd.ewma(df['MACD'], span=nSig)
	df['Signal Line Crossover'] = np.where(df['MACD'] > df['Signal Line'], 1, 0)
	df['Signal Line Crossover'] = np.where(df['MACD'] < df['Signal Line'], -1, df['Signal Line Crossover'])
	df['Centerline Crossover'] = np.where(df['MACD'] > 0, 1, 0)
	df['Centerline Crossover'] = np.where(df['MACD'] < 0, -1, df['Centerline Crossover'])
	df['Buy Sell'] = (2*(np.sign(df['Signal Line Crossover'] - df['Signal Line Crossover'].shift(1))))
	#df.plot(y=['price'], title='MACD Analysis')
	#df.plot(y= ['MACD', 'Signal Line'], title='MACD & Signal Line')
	#df.plot(y= ['Centerline Crossover', 'Buy Sell'], title='Signal Line & Centerline Crossovers', ylim=(-3,3))
	print >> sys.stderr, df.tail()
	return df
