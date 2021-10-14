#!/usr/bin/env python
import sys
import rpy2.robjects as robj
from rpy2.robjects import pandas2ri, r
import datetime
import pandas as pd
from _alan_calc import subDict,pull_stock_data
import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)

def run_fcs(ticker,debugTF=False,funcName='rForecast',**optx):
	# get data
	datax=pull_stock_data(ticker)
	asof=int(datax['pbdate'].iloc[-1])
#	idxtm=map(lambda x:datetime.datetime.strptime(str(x),"%Y%m%d"),datax['pbdate'])
#	datax.set_index(pd.DatetimeIndex(idxtm),inplace=True)
	if debugTF is True:
		print datax.tail()

	# get r-code
	pandas2ri.activate()
	rstring='source("./_alan_ohlc_fcs.r")'
	r(rstring)
	
	# convert to r-data 
	#df=pandas2ri.py2ri(datax[['pbdate','close']])
	df=pandas2ri.py2ri(datax['close'][:])

	# run r-function
	opts = {'nfcs': 30, 'dwmTF': True, 'autoArima': False, 'difTF': True, 'funcname': 'rAR', 'logTF': True, 'plevel': 0.7, 'freq': 'W'}
	opts.update(optx)
	optR = subDict(opts,['nfcs','plevel','funcname','autoArima','logTF','difTF','freq','fcsLst','dwmTF'])
	if debugTF:
		print >> sys.stderr, "==Input Args:{}".format(optR)
		print >> sys.stderr, "==asof {},df:\n{}".format(asof,datax['close'][-5:])
	if funcName in robj.globalenv:
		funcArg = robj.globalenv[funcName]
	ret=funcArg(df,asof,debugTF=debugTF,**optR)
        if opts['dwmTF'] is True:
                dwm=pandas2ri.ri2py(ret[1]);dwm['ticker']=ticker
        else:
                dwm=pd.DataFrame()
        dd=pandas2ri.ri2py(ret[0]);dd['ticker']=ticker
        return (dd,dwm,datax)

if __name__ == '__main__':
	args=sys.argv[1:]
	ticker = args[0]   if len(args)>0 else '^GSPC'
	funcName = args[1] if len(args)>1 else 'rForecast'
	sys.stderr.write("{}\n".format(run_fcs(ticker,funcName=funcName,debugTF=False)))
