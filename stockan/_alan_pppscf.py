#!/usr/bin/env python3 
""" program: _alan_pppfcs.py
    Description: ALAN past price performance summary curve fitting
    Example:
	python _alan_pppscf.py AAPL 
	python _alan_pppscf.py DGS10 --src=fred --category=macro
	python _alan_pppscf.py "USSTHPI|Q" --src=fred --category=macro
	printf "SELECT series,freq FROM mapping_series_label WHERE series<>label" | /apps/fafa/bin/psql.sh -d ara -At | python _alan_pppscf.py --src=fred --category=macro --table=macro_pppscf
    Version: 0.66
	def locate_mnmx(vy, locx=0, sign=1, scroll=5):
	def vertex_locator(dy, vx=None, span=5, npar=15, scroll=5, debugTF=False):
	def set_time_index(dg, dh, dtCol=None, parser=epoch_parser):
	def fq_switch(fq_str, fq_dict={'D':21,'W':12,'M':6,'Q':4,'Y':5}):
	def calc_pppchgs(dfr,zs=0.,stdev=1.,avg=0.,fqx='D'):
	def run_pppscf(ticker,opts,pgDB=None):
	def get_db_pg(dbname=None,hostname=None,port=5432):
	def batch_pppscf(tkLst,opts=None,optx=None):
	def opt_pppscf(argv,retParser=False):
    Note:
	find_polyroots() is replaced with vertex_locator()
    Last Mod., Tue Sep  4 18:09:01 EDT 2018
"""
import sys
from optparse import OptionParser
import pandas_datareader.data as web
from sqlalchemy import create_engine
import datetime
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from _alan_calc import get_csvdata,pullStockHistory,adjust_fq_days,pqint,ewma_smooth,subDict
from _alan_date import ymd_parser, epoch_parser

def locate_mnmx(vy, locx=0, sign=1, scroll=5):
	""" re-select tx locators to refine the min/max points
	"""
	nobs = len(vy)
	mnmx = max if sign >= 1 else (min if sign <= -1 else None)
	if mnmx is None:
		return locx
	start = max(locx - scroll, 0)
	end = min(locx + scroll, nobs)
	locy = mnmx(vy.iloc[start:end])
	vx = vy.iloc[start:end][vy == locy].index
	tx = vx[-1] if len(vx) else locx
	return tx


def vertex_locator(dy, vx=None, span=5, npar=15, scroll=5, debugTF=False):
	""" locate vertex of array [dy] via polyfit(npar) 
	    	optimize with ewma_smooth(span) and local MinMax locate_mnmx(scroll)  
		return dataframe dh[iptday,actual,fitted,sign]
		where
			span: for running w.moving average [span] period
			npar: for [npar]_th polynomial fitting over time
			scroll: for refine the min-max vertex point +- [scroll] period
			tx are the locators
			ty are the correspoding fitted values
			tz are the correspoding actual values
	"""
	nobs = len(dy)
	if isinstance(dy,pd.Series):
		vy = pd.Series(dy.values)
		vidx = dy.index
	else:
		vy = pd.Series(dy)
		vidx = vy.index
	if vx is None:
		vx = np.arange(nobs)
	vma = ewma_smooth(vy, span) if span > 1 else vy
	params = np.polyfit(vx, vma, npar)
	yfunc = np.poly1d(params)
	ydfunc = yfunc.deriv(m=1)
	vfit = np.polyval(params, vx)
	tx = [ int(x.real) for x in ydfunc.roots[~np.iscomplex(ydfunc.roots)] ]
	tx = list(filter(lambda x: x >= 0 and x <= nobs - 1, tx))
	tx = sorted(set(tx + [0, nobs - 1]))
	ty = list(map(lambda x: yfunc(x), tx))
	ty[0],ty[-1] = (vy.iloc[0], vy.iloc[-1])
	tz = list(vy[tx])
	tsg=[0]
	tsg+= [1 if x>0 else -1 if x <=0 else 0 for x in np.diff(ty)]
	dh=pd.DataFrame(list(zip(tx,ty,tz,tsg)),columns=['iptday','fitted','actual','sign'])

	# re-select tx locator (to refine min/max point
	if scroll > 0:
		tx[1:-1]=[locate_mnmx(vy,locx=x,sign=y,scroll=scroll) for x,y in zip(tx[1:-1],tsg[1:-1]) ]
		tx = sorted(set(tx))
		ty = [yfunc(x) for x in tx]
		ty[0],ty[-1] = (vy.iloc[0], vy.iloc[-1])
		tz = list(vy[tx])
		tsg[1:] = [1 if x>0 else -1 if x <=0 else 0 for x in np.diff(tz)]
		dh=pd.DataFrame(list(zip(tx,ty,tz,tsg)),columns=['iptday','fitted','actual','sign'])
	else:
			pqint( 'MNMX scroll not apply!', file=sys.stderr)
			dh[1:,'sign'] = [1 if x>0 else -1 if x <=0 else 0 for x in np.diff(tz)]
	dh['difday'] = dh['iptday'].diff().fillna(0).astype('int')
	idxname = vidx.name if vidx.name else 'date'
	dh[idxname] = vidx[dh['iptday'].values].values
	dh.index =  vidx[dh['iptday'].values]
	dh.index.name = idxname
	if debugTF is True:
		pqint( "dh in vertex_locator:\n{}".format(dh), file=sys.stderr)

	# no more interpolation, directly apply relative vertices w.r.t. index 
	dg = dh[['actual']]
	dg.columns = ['trend']

	if debugTF is True:
		pqint( "dg in vertex_locator:\n{}".format(dg), file=sys.stderr)
	return (dg, dh)

def set_time_index(dg, dh, dtCol=None, parser=epoch_parser):
	""" rearrange index as time-series
	"""
	dtIdx = list(map(parser, dtCol[dh['iptday']]))
	dh.set_index(pd.DatetimeIndex(dtIdx), inplace=True)
	dg.set_index(pd.DatetimeIndex(map(parser, dtCol)), inplace=True)
	return ( dg, dh)

def fq_switch(fq_str, fq_dict={'D':21,'W':12,'M':6,'Q':4,'Y':5}):
	""" return number of period based on one of specifid frequency strings:[D|W|M|Q|Y]
	"""
	return fq_dict[fq_str] if fq_str in fq_dict else fq_dict["D"]

def calc_pppchgs(dfr,zs=0.,stdev=1.,avg=0.,fqx='D'):
	"""['ticker','label','freq','pdate','xdate','iptday','difday','actual','fitted','slope',
	    'fit_dif','fit_chg','fit_chg_adj','act_dif','act_chg','zcore','fit_chg_zscore','act_chg_zscore',
	    'nthd','sig_daycount','sig_combo']
	"""
	# re-arrange dates 
	dfr['pdate']=[x.strftime("%Y-%m-%d") for x in dfr.date ]
	dfr['xdate']=dfr['pdate'].shift()
	dfr['pbdate']=[int(x.strftime("%Y%m%d")) for x in dfr.date ]

	# assign zs,stdev and avg
	nthd=fq_switch(fqx)
	dfr['avg_std']=zs;dfr['avg']=avg;dfr['stdev']=stdev

 	# calc additional variables
	dfr['fit_dif']=dfr['fitted'].diff();dfr['fit_chg']=dfr['fitted'].pct_change()
	dfr['act_dif']=dfr['actual'].diff();dfr['act_chg']=dfr['actual'].pct_change()
	dfr['slope']=dfr['act_dif']/dfr['difday']
	# Use [fit_chg*mean/std] for zscore to justify last period vaule of changing rate 
	# instead of directly applying: [fit_dif/std]
	dfr['fit_chg_zscore']=dfr['fit_chg']*zs;dfr['act_chg_zscore']=dfr['act_chg']*zs
	mx = dfr['difday'].max();dfr['sig_difday']= list(map(lambda x: 1 if x>=mx else 0,dfr['difday']))
	mx = dfr['fit_chg'].abs().max();dfr['sig_fit_chg']= list(map(lambda x: 1 if abs(x)>=mx else 0,dfr['fit_chg']))
	mx = dfr['act_chg'].abs().max();dfr['sig_act_chg']= list(map(lambda x: 1 if abs(x)>=mx else 0,dfr['act_chg']))
	dfr=dfr.fillna(0) # filling NA with 0 OR backward-filling fillna(method='bfill')
	return dfr

def vertex_mnmx(dh,prc_open,colx='actual'):
	if colx not in dh.columns:
		return {}
	dh.loc[dh.index[0],colx]=prc_open
	dmx = dh.loc[dh[colx] == dh[colx].max()].iloc[0].to_dict()
	dmn = dh.loc[dh[colx] == dh[colx].min()].iloc[0].to_dict()
	dk = []
	if dmx['iptday']>dmn['iptday']:
		dk = dk + [dmn, dmx]
	else:
		dk = dk + [dmx, dmn]
	if dk[0]['iptday'] > dh['iptday'].iloc[0]: 
		dk = [dh.iloc[0].to_dict()] + dk
	if dk[-1]['iptday'] < dh['iptday'].iloc[-1]: 
		dk = dk + [dh.iloc[-1].to_dict()]	
	dk = pd.DataFrame(dk)
	dk['pchg_open'] = [ x/prc_open-1 for x in dk[colx]]
	dk['sign'] = [ 1 if x>0 else -1  for x in dk['pchg_open']]
	dk['difday'] = dk['iptday'].diff()
	return dk

def run_pppscf(ticker,opts=None,pgDB=None,stdinTF=False,debugTF=False):
	"""
	Calc Past Price Performance Summary & CashFlow
	Return dictionary of {'data','dfdr'}
	where
		data: dataframe data of historical prices applied
		dfdr: dataframe the PPP Summary of data
	"""
	if opts is None:
		opts, _ = opt_pppscf([])
	#python3 compatible
	#for ky,va in opts.items():
	#	exec("{}=va".format(ky))
	npar = opts['npar']
	optx = subDict(opts,['sep','start','end','src','days'])
	if stdinTF is True:
		pqint( "==RUNNING get_csvdata({sep},{start},{end},{src},{days})".format(**optx), file=sys.stderr)
		data=get_csvdata('-',pgDB=pgDB,**optx)
	else:
		data=pullStockHistory(ticker,pgDB=pgDB,**optx)
	if isinstance(data.index, (datetime.date, datetime.datetime)) is False:
		if 'pbdate' in data:
			idxpt=[ymd_parser(x,fmt="%Y%m%d") for x in data['pbdate']]
			data.set_index(pd.DatetimeIndex(idxpt),inplace=True)
		elif 'epochs' in data:
			idxpt=[epoch_parser(x) for x in data['epochs']]
			data.set_index(pd.DatetimeIndex(idxpt),inplace=True)
	prc=data['close']
	prc=prc.dropna(axis=0, how='all')

	# polyfit the data at [npar] polynomial
	_, dfdr = vertex_locator(prc,npar=npar,debugTF=debugTF)

	pqint( dfdr, file=sys.stderr)
	pqint( data.head(5), file=sys.stderr)
	# calc additional stats 
	vx=prc[-62:] # vx=prc # using last 3-month as z-score benchmark rather than entire period
	stdev=vx.std();avg=vx.mean();zs=avg/stdev
	dfdr=calc_pppchgs(dfdr,zs=zs,stdev=stdev,avg=avg)
	dfdr=dfdr.drop(['date'],1)
	dfdr['ticker']=ticker
	return {"data":data,"dfdr":dfdr}

def get_db_pg(dbname=None,hostname=None,port=5432):
	engine=None
	if not (dbname is None or hostname is None):
		try:
			dbURL='postgresql://sfdbo@{}:{}/{}'.format(hostname,port,dbname)
			engine = create_engine(dbURL) if(dbURL is not None) else None
		except:
			pqint( "***DB ERROR:", sys.exc_info()[1], file=sys.stderr)
	return engine

def batch_pppscf(tkLst,sector='Technology',va=None,j2ts=None,mp3YN=False,category='stock',wmode='replace',end=None,debugTF=False,hostname='localhost',tablename='ohlc_pppscf',saveDB=False,start=None,j2name=None,freq='D',npar=15,dbname='ara',lang='cn',src='yahoo',plotFlg=False,days=730,output=None,ky='pngname',pngname=None,sep='|',fpTF=False,**optx):
	from lsi_daily import run_comment_pppscf #- test ONLY
	opts=locals()
	opts.pop('tkLst',None)
	opts.pop('optx',None)
	if optx is not None:
		opts.update(optx)
	if opts['category'] != 'stock':
		opts['src'] = 'fred'
	#for ky,va in opts.items():
	#	exec("{}=va".format(ky))
	if debugTF is True:
		pqint("===>", opts, file=sys.stderr)
	pgDB=get_db_pg(dbname=dbname,hostname=hostname)
	wmode='replace'
	dcmt=''
	stdinTF = False
	for j,ticker in enumerate(tkLst):
		if '-' in ticker:
			ticker,_= ticker.split(',')
			stdinTF = True
		if '|' in ticker:
			alst = ticker.split('|')
			dlst = [alst[0],freq,sector,alst[0]]
			(ticker,freq,sector,label) = alst+dlst[len(alst)-len(dlst):]
		else:
			label = ticker
		if freq != 'D':
			nday=adjust_fq_days(days,fq=freq)
			opts['freq']=freq
			opts['days']=nday
		else:
			nday=days
		if debugTF is True:
			pqint( "days:{} after {} adjust.".format(nday,freq), file=sys.stderr)
			pqint( "===RUNNING {}".format(ticker), file=sys.stderr)
		try:
			ret = run_pppscf(ticker,opts,pgDB=pgDB,stdinTF=stdinTF,debugTF=debugTF)
			df=ret['dfdr']
			if debugTF is True:
				pqint( ret['data'].tail(2) , file=sys.stderr)
			if saveDB is True:
				df.to_sql(tablename, pgDB, schema='public', index=False, if_exists=wmode)
				wmode='append'
			else:
				if debugTF is True:
					pqint( df.to_csv(sep="|"), file=sys.stderr)
				# for lang='cn' case ONLY
				if category == 'stock': 
					xqTmp="SELECT ticker,company{0} as label,'D' as freq,'stock' as category, sector{0} as sector FROM mapping_ticker_cik WHERE ticker='{1}'"
				else:
					xqTmp="SELECT series as ticker,label{0} as label, freq,category, category{0} as sector FROM mapping_series_label WHERE series='{1}'"

				dx=pd.read_sql(xqTmp.format("_cn" if lang=="cn" else "", ticker),pgDB)
				if dx.shape[0]<1:
					dx=pd.DataFrame([dict(ticker=ticker,label=label,freq=freq,category=category,sector=sector)])
				fp=df.merge(dx,on='ticker')
				j2ts=j2ts if j2ts is not None else open(j2name).read() if j2name is not None else None
				dcmt = run_comment_pppscf(ticker=ticker,fp=fp,pgDB=pgDB,lang=lang,ts=j2ts)
				pqint(dcmt, file=sys.stdout)
		except Exception as e:
			pqint( '**ERROR {} [{}]:{}'.format("batch_pppscf()",ticker,str(e)), file=sys.stderr)
			continue

	if pgDB is not None:
		pgDB.dispose()
	global gData
	try:
		from lsi_daily import gData
	except:
		gData={}
	return dcmt if fpTF is False else (dcmt, fp)

def opt_pppscf(argv,retParser=False):
	""" command-line options initial setup
	    Arguments:
		argv:   list arguments, usually passed from sys.argv
		retParser:      OptionParser class return flag, default to False
	    Return: (options, args) tuple if retParser is False else OptionParser class
	"""
	parser = OptionParser(usage="usage: %prog [option] SYMBOL1 ...", version="%prog 0.65",
		description="ALAN Past Price Performance Summary Curve Fitting") 
	parser.add_option("-s","--sep",action="store",dest="sep",default="|",
		help="field separator (default: |)")
	parser.add_option("","--start",action="store",dest="start",
		help="start YYYY-MM-DD (default: 3-years-ago)")
	parser.add_option("","--end",action="store",dest="end",
		help="end YYYY-MM-DD (default: today)")
	parser.add_option("","--days",action="store",dest="days",default=730,type=int,
		help="number of days from endDate (default: 730)")
	parser.add_option("-d","--database",action="store",dest="dbname",default="ara",
		help="database (default: ara)")
	parser.add_option("","--host",action="store",dest="hostname",default="localhost",
		help="db host (default: localhost)")
	parser.add_option("-t","--table",action="store",dest="tablename",default="ohlc_pppscf",
		help="db tablename (default: ohlc_pppscf)")
	parser.add_option("-w","--wmode",action="store",dest="wmode",default="replace",
		help="db table write-mode [replace|append|fail] (default: replace)")
	parser.add_option("-o","--output",action="store",dest="output",
		help="output type (default: None)")
	parser.add_option("","--src",action="store",dest="src",default="yahoo",
		help="source [fred|yahoo](default: yahoo)")
	parser.add_option("","--category",action="store",dest="category",default="stock",
		help="category [stock|macro|interest_rate](default: stock)")
	parser.add_option("","--freq",action="store",dest="freq",default="D",
		help="frequency period [D|W|M|Q|Y](default: D)")
	parser.add_option("","--sector",action="store",dest="sector",default="Technology",
		help="sector (default: Technology)")
	parser.add_option("","--npar",action="store",dest="npar",default=15,type="int",
		help="nth polynomial (default: 15)")
	parser.add_option("","--plot",action="store_true",dest="plotFlg",default=False,
		help="show plot(default: False)")
	parser.add_option("","--no_database_save",action="store_false",dest="saveDB",default=True,
		help="no save to database (default: save to database)")
	parser.add_option("-l","--lang",action="store",dest="lang",default="cn",
		help="db language mode [cn|en] (default: cn)")
	parser.add_option("","--png",action="store",dest="pngname",
		help="graph name (default: None)")
	parser.add_option("","--j2ts",action="store",dest="j2ts",
		help="jinja2 template script, (default: None).")
	parser.add_option("","--j2name",action="store",dest="j2name",
		help="jinja2 template file (default: None). ONLY valid if j2ts is None")
	parser.add_option("","--use_mp3",action="store_true",dest="mp3YN",default=False,
		help="comment use mp3 style")
	parser.add_option("","--debug",action="store_true",dest="debugTF",default=False,
		help="debugging (default: False)")
	(options, args) = parser.parse_args(argv[1:])
	if retParser is True:
		return parser
	return (vars(options), args)

if __name__ == '__main__':
	(opts, args)=opt_pppscf(sys.argv)
	if len(args)==0 or args[0]=='-' :
		tkLst = sys.stdin.read().strip().split("\n")
	else :
		tkLst = args
		opts['saveDB']=False
	batch_pppscf(tkLst, **opts)
