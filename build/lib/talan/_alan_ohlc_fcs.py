#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" program: _alan_ohlc_fcs.py
    Description: ALAN Day, Week, Month Forecast
    Version: 0.66
    Example:
	python _alan_ohlc_fcs.py AAPL 
	python _alan_ohlc_fcs.py DGS10 --src=fred --category=macro
	printf "SELECT series,freq FROM mapping_series_label WHERE series<>label" | /apps/fafa/bin/psql.sh -d ara -At | python _alan_ohlc_fcs.py --src=fred --category=macro --table=macro_fcs --no_dif
    Functions:
	def run_ohlc_fcs(ticker,opts=None,debugTF=False,pgDB=None,**kwargs):
	def batch_ohlc_fcs(tkLst,opts):
	def opt_ohlc_fcs(argv,retParser=False):
    Growth rate's are used for (GDPs,HPI, CPI & PPIs)
    lnLst=c("A939RX0Q048SBEA_PCTCHG","GDP_PCTCHG","HPIPONM226S_PCTCHG","CPIAUCNS_PCTCHG","WPUFD49207_PCTCHG","PPIACO_PCTCHG")
    Last Mod., Wed Sep 19 13:46:00 EDT 2018
"""
import sys,os
from optparse import OptionParser
import rpy2.robjects as robj
from rpy2.robjects import pandas2ri, r
import datetime
import numpy as np
import pandas as pd
from _alan_calc import pull_stock_data,conn2pgdb,adjust_fq_days,subDict,pqint,getKeyVal
from lsi_daily import run_comment_fcst #- test ONLY
import warnings
from rpy2.rinterface import RRuntimeWarning
warnings.filterwarnings("ignore", category=RRuntimeWarning)


def run_ohlc_fcs(ticker,opts=None,debugTF=False,pgDB=None,**kwargs):
	""" 
	forecast 'nfcs' periods based on raw data 'datax' 
	return (dd,dwm,datax)
	where
		dd: forecast values
		dwm: forecast values of next day, week, month (optional)
		datax: data used for forecast calculation
	Note, dwm can be optional if dwmTF is False

	"""
	if opts is None:
		(opts, _)=opt_ohlc_fcs([])
	if len(kwargs)>0:
		opts.update(kwargs)
	if debugTF:
		pqint( opts, file=sys.stderr)
	days = getKeyVal(opts,'days',730)
	freq = getKeyVal(opts,'freq','D')
	# get data
	if isinstance(ticker,pd.DataFrame):
		datax=ticker
		ticker = ''
	else: # get data
		optx = subDict(opts,['src','days','start','end'])
		datax=pull_stock_data(ticker,pgDB=pgDB,**optx)

	if 'ticker' in datax:
		ticker=datax['ticker'].iloc[0]

	if datax is None or len(datax)<1:
		return (None,None,None)
	#idxtm=map(lambda x:datetime.datetime.strptime(str(x),"%Y%m%d"),datax['pbdate'])
	#datax.set_index(pd.DatetimeIndex(idxtm),inplace=True)
	if debugTF is True:
		pqint( opts, file=sys.stderr)
		pqint( datax.tail(), file=sys.stderr)
	nobs=days
	if 'epochs' in datax:
		asof = int(datetime.datetime.fromtimestamp(int(datax['epochs'].iloc[-1])/1000).strftime('%Y%m%d'))
		fcsLst=np.array([5,10,30])
	else:
		asof=int(datax['pbdate'].iloc[-1])
		fcsLst=np.array([1,5,23])
	vprc=datax['close'][-nobs:]

	# get r-code
	pandas2ri.activate()
	fpath = os.path.dirname(__file__)
	if len(fpath)<1:
		fpath = '.' 
	rstring='source("{}/{}")'.format(fpath, "_alan_ohlc_fcs.r")
	if debugTF:
		pqint( rstring, file=sys.stderr)
	r(rstring)

	# convert to r-data
	df=pandas2ri.py2ri(vprc)

	# run r-function [rGARCH|rAR]
	optx = subDict(opts,['nfcs','plevel','funcname','autoArima','logTF','difTF','freq','fcsLst','dwmTF'])
	if debugTF:
		pqint( "==Input Args:{}".format(optx), file=sys.stderr)
		pqint( "==df\n:{}".format(vprc.tail()), file=sys.stderr)
	ret=robj.globalenv['rForecast'](df,asof,debugTF=debugTF,**optx)
	#ret=robj.globalenv['rForecast'](df,asof,plevel=plevel,funcname=funcname,autoArima=autoArima,debugTF=debugTF,logTF=logTF,difTF=difTF,freq=freq,fcsLst=fcsLst)
	if opts['dwmTF'] is True:
		dwm=pandas2ri.ri2py(ret[1])
		dwm['ticker']=ticker
	else:
		dwm=pd.DataFrame()
	dd=pandas2ri.ri2py(ret[0])
	dd['ticker']=ticker
	dd['freq']=freq
	return (dd,dwm,datax)

def convert_data_comment_fcst(ticker,category,df,pgDB=None,lang="cn",mp3YN=True,ts=None,fpTF=False):
	if category == 'stock':
		xqTmp="SELECT ticker,company{0} as label,'D' as freq,'stock' as category, sector{0} as sector FROM mapping_ticker_cik WHERE ticker='{1}'"
		prcn=0
	else:
		xqTmp="SELECT series as ticker,label{0} as label, freq,category, category{0} as sector FROM mapping_series_label WHERE series='{1}'"

		prcn=2

	if lang == "en":
		if ts==None:
			ts="{label} closed {xdTrendWd} {xdChgWd} at {price} {xdayWd}, {xwTrendWd} of {xwChgWd} for the week. This {xwChgWd} {movingWd} is {cmpWd} the historical volatility of {sigmaWd}. Our {label} forecast for the next week is {nwTrendWd} with a probability of {posPbWd}, and {plevelWd} chance of closing {rangeWd}."
		dotSign=' point ' if mp3YN is True else '.'
	else:
		if ts==None:
			tsClosing= "目前收盤價{price}元" if category == "stock" else "目前為{price}"
			ts="{label}在前一{unitStr}{pastTrendWd} {xwChgWd}，"+tsClosing+" 。這個{xwChgWd} {movingWd}{cmpWd} 之前的歷史波動率{sigmaWd}。依據{label}波動狀況，預估下一{unitStr}有七成可能{rangeWd}。{posPbWd}"
		dotSign='點' if mp3YN is True else '.'
		#ts=None

	#- NOTE: pandas series need to re-cast as dictionary again
	try:
		fx=pd.read_sql(xqTmp.format("_cn" if lang=="cn" else "", ticker),pgDB).iloc[0].to_dict()
	except:
		fx=dict(ticker=ticker,label=ticker,freq='D',category='stock',sector='Technology')
	if fx['freq']=="D":
		fp=(df.loc[df['freq']=="W"]).iloc[0].to_dict()
		fp.update(fx)
		fp['freq']="W"
	else:
		fp=(df.loc[df['freq']=="D"]).iloc[0].to_dict()
		fp.update(fx)
	dcmt = run_comment_fcst(ticker=ticker,fp=fp,pgDB=pgDB,prcn=prcn,ts=ts,lang=lang,dotSign=dotSign,mp3YN=mp3YN)
	return dcmt if fpTF is False else (dcmt, fp)

def batch_ohlc_fcs(tkLst,opts=None,optx=None,fpTF=False):
	if opts is None:
		opts, _ = opt_ohlc_fcs([])
	if optx is not None:
		opts.update(optx)
	if opts['category'] != 'stock':
		opts['src'] = 'fred'
	for ky,va in opts.items():
		exec("{}=va".format(ky))
	debugTF = getKeyVal(opts,'debugTF',False)
	dbname = getKeyVal(opts,'dbname',False)
	hostname = getKeyVal(opts,'hostname','localhost')
	days = getKeyVal(opts,'days',730)
	freq = getKeyVal(opts,'freq','D')
	j2ts = getKeyVal(opts,'j2ts',None)
	j2name = getKeyVal(opts,'j2name',None)
	category = getKeyVal(opts,'category','stock')
	lang = getKeyVal(opts,'lang','cn')
	saveDB = getKeyVal(opts,'saveDB',False)
	mp3YN = getKeyVal(opts,'mp3YN',False)
	if debugTF is True:
		pqint( opts, file=sys.stderr)
	pgDB=conn2pgdb(dbname=dbname,hostname=hostname)
	dcmt=''
	for ticker in tkLst:
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
		try:
			(dd,df,datax) = run_ohlc_fcs(ticker,opts,pgDB=pgDB)
			if df is None:
				continue
			pqint( df.tail(3), file=sys.stderr)
			j2ts=j2ts if j2ts is not None else open(j2name).read() if j2name is not None else None
			dcmt, fp = convert_data_comment_fcst(ticker,category,df,pgDB,lang=lang,mp3YN=mp3YN,ts=j2ts,fpTF=True)
			if saveDB is True:
				df.to_sql(tablename, pgDB, schema='public', index=False, if_exists=wmode)
				wmode='append'
				pqint( df, file=sys.stderr)
			else:
				pqint( df.to_csv(sep="\t"), file=sys.stderr)
			pqint( dcmt, file=sys.stdout)
		except Exception as e:
			pqint( str(e), file=sys.stderr)
			pqint( '**ERROR {}: {}'.format("batch",ticker), file=sys.stderr)
			continue

	if pgDB is not None:
		pgDB.dispose()
	global gData
	from lsi_daily import gData
	return dcmt if fpTF is False else (dcmt, fp)

def opt_ohlc_fcs(argv,retParser=False):
	""" command-line options initial setup
	    Arguments:
		argv:   list arguments, usually passed from sys.argv
		retParser:      OptionParser class return flag, default to False
	    Return: (options, args) tuple if retParser is False else OptionParser class
	"""
	parser = OptionParser(usage="usage: %prog [option] SYMBOL1 ...", version="%prog 0.65",
		description="ALAN Day, Week, Month Forecast") 
	parser.add_option("-s","--start",action="store",dest="start",
		help="start YYYY-MM-DD (default: 3-years-ago)")
	parser.add_option("-e","--end",action="store",dest="end",
		help="end YYYY-MM-DD (default: today)")
	parser.add_option("","--days",action="store",dest="days",default=730,type=int,
		help="number of days from endDate (default: 730)")
	parser.add_option("-d","--database",action="store",dest="dbname",default="ara",
		help="database (default: ara)")
	parser.add_option("","--host",action="store",dest="hostname",default="localhost",
		help="db host (default: localhost)")
	parser.add_option("-t","--table",action="store",dest="tablename",default="ohlc_fcs",
		help="db tablename (default: ohlc_fcs)")
	parser.add_option("","--funcname",action="store",dest="funcname",default="rAR",
		help="forecast function [rAR|rGARCH]  (default: rAR)")
	parser.add_option("","--autoArima",action="store_true",dest="autoArima",default=False,
		help="use autoArima order based on AIC criteria when rAR is applied (default: False)")
	parser.add_option("","--src",action="store",dest="src",default="yahoo",
		help="source [fred|yahoo](default: yahoo)")
	parser.add_option("","--category",action="store",dest="category",default="stock",
		help="category [stock|macro|interest_rate](default: stock)")
	parser.add_option("","--freq",action="store",dest="freq",default="D",
		help="frequency period [D|W|M|Q|Y](default: D)")
	parser.add_option("","--nfcs",action="store",dest="nfcs",default=30,type=int,
		help="number of forecast periods (default: 30)")
	parser.add_option("","--no_dwm",action="store_false",dest="dwmTF",default=True,
		help="Retrive next day/week/month forecast values (default: True)")
	parser.add_option("","--sector",action="store",dest="sector",default="Technology",
		help="sector (default: Technology)")
	parser.add_option("-w","--wmode",action="store",dest="wmode",default="replace",
		help="db table write-mode [replace|append|fail] (default: replace)")
	parser.add_option("","--plevel",action="store",dest="plevel",default=0.7,type=float,
		help="forecast confidence interval in decimal (default: 0.7)")
	parser.add_option("","--no_database_save",action="store_false",dest="saveDB",default=True,
		help="no save to database (default: save to database)")
	parser.add_option("","--no_log",action="store_false",dest="logTF",default=True,
		help="Not apply [log] form to data (default: True)")
	parser.add_option("","--no_diff",action="store_false",dest="difTF",default=True,
		help="Not apply 1st-differece data (default: True)")
	parser.add_option("-l","--lang",action="store",dest="lang",default="cn",
		help="db language mode [cn|en] (default: cn)")
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
	(options, args)=opt_ohlc_fcs(sys.argv)
	if len(args)==0 or args[0]=='-' :
		pqint("\nRead from pipe\n\n", file=sys.stderr)
		tkLst = sys.stdin.read().strip().split("\n")
	else:
		tkLst = args
		options['saveDB']=False
	batch_ohlc_fcs(tkLst,options)
	pqint( os.path.dirname(__file__), file=sys.stderr)
