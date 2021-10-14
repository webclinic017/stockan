#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: Technical analysis utility functions
Version: 0.71
Last mod., Fri Apr 12 14:32:24 EDT 2019
"""
import sys
from optparse import OptionParser
import numpy as np
import pandas as pd
import datetime
import json
import re
#%matplotlib inline 
#import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d
pd.options.display.float_format='{:,.2f}'.format

def saferun(func):
	'''
	Decorator for try_except to wrap func(...) 
	Error return is an empty string ""

	Usage of,
	@saferun 
	def func(...):
	----------------------------------------------------------------------
	'''
	def wrapped_f(*args, **kwargs):
		ret=""
		try:
			return func(*args, **kwargs)
		except Exception as e:
			sys.stderr.write("**ERROR: {},\nreturn {}\n".format(str(e),ret))
			return ret
	return wrapped_f

class safeRunArg(object):
	'''
	Decorator for try_except to wrap func(...) 

	Default error return is None

	Usage of,
	@safeRunArg()
	def func(...):

	OR
	@safeRunArg('WARNINGS') 
	def func(...):
	----------------------------------------------------------------------
	'''
	#if sys.version_info.major == 2:
	#	from singledispatch import singledispatch
	#	from functools import update_wrapper
	#else:
	#	from functools import singledispatch, update_wrapper

	def __init__(self, arg1=None):
		"""
		If there are decorator arguments, the function
		to be decorated is not passed to the constructor!
		"""
		self.arg1 = arg1

	def __call__(self, func,*args, **kwargs):
		"""
		If there are decorator arguments, __call__() is only called
		once, as part of the decoration process! You can only give
		it a single argument, which is the function object.
		"""
		#dispatcher = singledispatch(func)
		def wrapped_f(*args, **kwargs):
			try:
				return func(*args, **kwargs)
			except Exception as e:
				errMsg="**ERROR: {}\n{}".format(str(e),self.arg1)
				sys.stderr.write(errMsg+"\n")
				return self.arg1
		#wrapped_f.register = dispatcher.register
		#update_wrapper(wrapped_f, func)
		return wrapped_f

def printerr(*args,**kwargs):
	file=kwargs.pop('file',sys.stderr)
	return pqint(*args,file=file,**kwargs)

def pqint(*args,**kwargs):
	file = kwargs['file'] if 'file' in kwargs else sys.stdout
	sep = kwargs['sep'] if 'sep' in kwargs else ' '
	end = kwargs['end'] if 'end' in kwargs else '\n'
	flush = kwargs['flush'] if 'flush' in kwargs else False
	for x in ['file','sep','end','flush']:
		kwargs.pop(x,None)
	ret=0
	try:
		for j,x in enumerate(args):
			if j>0:
				file.write(sep)
			file.write("{}".format(x))
			ret +=1
		if len(args)>0:
				file.write(sep)
		for j,(k,v) in enumerate(kwargs.items()):
			if j>0:
				file.write(sep)
			file.write("{}".format(v))
			ret +=1
		if len(args)+len(kwargs)>0:
			file.write(end)
		if flush is True:
			file.flush()
	except Exception as e:
		sys.stderr.write(str(e)+"\n")
	return ret

def getKeyVal(opts={},key='',val=None,extendTF=False):
	"""
	Assign value from dict 'opts'['key'] to variable 'key' with default 'val'
	Example:
	  saveDB=getKeyVal(opts,key='saveDB',val=True)
	Note
	  1. both 'key' and 'val' can be a list for multi-assignments
	  2. if extendTF is True and the 'key' list is greater than 'val' list, 
		 None will be assigned for additional 'key' 
	"""
	if key is None or len(key)<1:
		return val
	if isinstance(key,list):
		if isinstance(val,list):
			if extendTF is True:
				val.extend([None]*(len(key)-len(val)))
		else:
			val.extend([None]*len(key))
		return [getKeyVal(opts,k,v) for k,v in zip(key,val)]
	else:
		return opts[key] if key in opts else val

def pdGroupMax(df=[],by=[],axis='',func='max',indexOnly=False):
	"""
	Get dataframe groupby 'by' and select 'func' of colummn 'axis'
	Example,
	pdGroupMax(df,by=['ticker'],axis='epochs',func='max')
	"""
	if not all([len(df)>0,len(by)>0,len(axis)>0]):
		return []
	idx = df.groupby(by)[axis].transform(func)==df[axis]
	return df.loc[idx] if indexOnly is False else idx

def getCurrHist(file=sys.stdout):
	"""
	print python interative mode history
	"""
	import readline
	h=[]
	for i in range(readline.get_current_history_length()):
		htmp = readline.get_history_item(i+1)+"\n"
		file.write(htmp)
		h.append(htmp)
	return h
	
def renameDict(dc,mapper={},copyTF=False):
	"""
	rename dictionary keys of 'dc' based on 'mapper'
	where
	  mapper is dict-like to transform the old key to new one
	  return a new copy of 'dc' if copyTF is True
	"""
	if not mapper:
		return dc
	if isinstance(dc,list):
		dM = [ renameDict(d,mapper,copyTF) for d in dc]
		return dM
	elif isinstance(dc,pd.DataFrame):
		ks = mapper.keys()
		xcol = dc.columns
		M = {ky:va for ky,va in mapper.items() if ky in xcol}
		dc.rename(columns=M,inplace=True)
		return dc
	M=dc.copy() if copyTF else dc
	for ky,va in mapper.items():
		if ky not in M or ky == va:
			continue
		M[va] = M.pop(ky)
	return M

def subDF(M,K=[],reverseTF=False):
	""" 
	return a subset of dataframe 'M' that matches 'K' columns
	OR
	if reverse is True
	a subset of M that does not match K columns 
	"""
	if reverseTF is True: # invert-match, select non-matching [kyLst] keys 
		colLst = [ky for ky in M.columns if ky not in K]
	else:
		colLst = [ky for ky in K if ky in M.columns]
	return M[colLst]

def subDict(M,K=[],reverseTF=False):
	""" 
	return a subset of M that matches K keys
	OR
	if reverse is True
	a subset of M that does not match K keys 
	"""
	if isinstance(M,pd.DataFrame):
		return subDF(M,K,reverseTF=reverseTF)
	if isinstance(M,list):
		dM = [ subDict(d,K,reverseTF) for d in M]
		return dM
	if reverseTF is True: # invert-match, select non-matching [kyLst] keys 
		return { ky:va for ky,va in M.items() if ky not in K }
	else:
		return { ky:va for ky,va in M.items() if ky in K }

def subVDict(M,V=[None],reverseTF=False):
	"""
	return a subset of M that matches V values
	OR
	if reverse is True
	return a subset of M that does not match V values
	"""
	if reverseTF is True: # invert-match, select non-matching [kyLst] keys 
		return {ky:va for ky,va in M.items() if va not in V}
	else:
		return {ky:va for ky,va in M.items() if va in V}

def step_index(a):
	''' return indices of a[] whenever the value changes
	'''
	b=[]
	if len(a)<1:
		return b
	z=a[0]
	for j,x in enumerate(a):
		if z != x:
			z=x
			b.append(j)
	return b

def chk_sign(a,b):
	if a*b>=0:
		xs = 1 if abs(a)>abs(b) else -1
	else:
		xs = 0
	return xs

def extrapolate_series(yo):
	yg=yo.dropna()
	fn = interp1d(map(int,yg.index.values), yg.values, fill_value='extrapolate')
	return fn(map(int,yo.index.values))

def sqlQuery(sql=None,engine=None,dbname="ara",hostname="localhost",port=5432,dialect='postgresql',driver='',user='sfdbo',dbTF=False,parse_dates=None):
	"""
	return DataFrame of SQL statement 'sql' based on SQL connnection 'engine'
	Note, if database connection 'dbTF' is True
	  return df,engine 
	"""
	if sql is None:
		return []
	engine = conn2db(engine=engine,dbname=dbname,hostname=hostname,port=port,dialect=dialect,driver=driver,user=user)
	if engine is None:
		return []
	df = pd.read_sql(sql,engine)
	if isinstance(parse_dates, dict):
		df = df.set_index(parse_dates.keys())
	return df if dbTF is False else (df,engine)
	
def conn2db(engine=None,dbname=None,hostname="localhost",port=5432,dialect='postgresql',driver='',user='sfdbo',dbURL=''):
	"""
		dialect=[postgresql,mysql,sqlite,mssql,mongo]
	"""
	from pymongo import MongoClient
	from sqlalchemy import create_engine
	if engine is not None: 
		return engine
	try:
		if dbURL:
			engine = create_engine(dbURL)
		elif all([hostname,port]) and dialect.lower() in ['mongo','mongodb']:
			if isinstance(dbname,str):
				dbname = dbname.replace('.','_')
				host='{}:{}'.format(hostname,port)
				engine = MongoClient(host)[dbname]
			else:
				host='{}:{}'.format(hostname,port)
				engine = MongoClient(host)
		elif all([hostname,port,dbname]):
			dbURL='{}{}://{}@{}:{}/{}'.format(dialect,driver,user,hostname,port,dbname)
			engine = create_engine(dbURL)
		pqint( "===DB-Driver:",engine, file=sys.stderr)
	except:
		pqint( "***DB ERROR:", sys.exc_info()[1], file=sys.stderr)
	return engine

def conn2mydb(engine=None,dbname=None,hostname="localhost",port=3306,dialect='mysql',driver='',user='sfdbo'):
	return conn2db(engine=engine,dbname=dbname,hostname=hostname,port=port,dialect=dialect,driver=driver,user=user)

def conn2pgdb(engine=None,dbname=None,hostname="localhost",port=5432,dialect='postgresql',driver='',user='sfdbo'):
	return conn2db(engine=engine,dbname=dbname,hostname=hostname,port=port,dialect=dialect,driver=driver,user=user)

def conn2mgdb(engine=None,dbname=None,hostname="localhost",port=27017,dialect='mongo'):
	return conn2db(engine=engine,dbname=dbname,hostname=hostname,port=port,dialect=dialect)

def save2pgdb(df,db=None,tablename="temp",wmode='replace',debugTF=False):
	if db is None:
		return None
	if isinstance(db,str):
		db = conn2pgdb(dbname=db)
	if debugTF:
		sys.stderr.write("{} of {}\n".format(tablename,db))
	df.to_sql(tablename, db, schema='public', index=False, if_exists=wmode)

def upd_temp2hist(pgDB=None,temp=None,hist=None,pcol=[],dbname=None,hostname='localhost',df={}):
	"""
	Insert/update additional values from table: [temp] to [hist]
	base on primary keys pcol
	"""
	if any(x==None for x in (temp,hist)) is True:
		return None
	xqTmp='''CREATE TABLE IF NOT EXISTS "{hist}" AS SELECT * FROM "{temp}" WHERE 1=2; {delC} 
INSERT INTO "{hist}" SELECT DISTINCT * FROM "{temp}"'''
	if len(pcol)>0:
		whrC = 'WHERE '+' AND '.join(['B."{0}" = C."{0}"'.format(j) for j in pcol])
		delC = '\nDELETE FROM "{hist}" B USING "{temp}" C {whrC} ;'.format(hist=hist,temp=temp,whrC=whrC)
	else:
		delC = ''
	xqr = xqTmp.format(hist=hist,temp=temp,delC=delC)
	try:
		if all(x==None for x in (pgDB,dbname)) is True:
			pqint("**ERROR: DB not defined!", file=sys.stderr)
			return xqr
		elif pgDB is None and dbname is not None:
			pgDB=conn2pgdb(dbname=dbname,hostname=hostname)
		if len(df)>0:
			df.to_sql(temp, pgDB, schema='public', index=False, if_exists='replace')
			pqint("Save {} to {}::{}".format(df.tail(1),dbname,temp), file=sys.stderr)
		pgDB.execute(xqr,pgDB)
	except Exception as e:
		return None, str(e)
	return pgDB,xqr

def adjust_fq_days(d=1090,fq='D',fqDct={'D':1,'W':2,'M':5,'Q':12,'Y':30}):
	mx=fqDct.get(fq.upper())
	return (d*mx) if mx>0 else d

def get_start_end(start=None,end=None,days=1090):
	""" 
	Description: START, END and DAYS ago adjustment
	Return: (start, end) in datetime format
	Parameters:
		start: START date, default to (END - DAYS)
		end: END date, default to NOW
		days: DAYS ago from END date, default to 1090
	"""
	if end is None:
		end = datetime.datetime.now()
	else:
		if isinstance(end,(int,np.integer,float)): 
			end=str(int(end))
		xfmt = '%Y%m%d' if end.isdigit() is True else '%Y/%m/%d' if '/' in end else '%Y-%m-%d'
		end=datetime.datetime.strptime(end,xfmt)
	if start is None:
		start = end - datetime.timedelta(days=days) 
	else:
		if isinstance(start,(int,np.integer,float)): 
			start=str(int(start))
		xfmt = '%Y%m%d' if start.isdigit() is True else '%Y/%m/%d' if '/' in start else '%Y-%m-%d'
		start = datetime.datetime.strptime(start,xfmt)
	return (start,end)

def hdr_data_stock(symbol,df):
	if df is None:
		return df
	df.columns=map(lambda x: x.lower().replace('adj close','adjusted'),df.columns)
	if 'adjusted' not in df.columns:
		df['adjusted'] = df['close']
	df=df.dropna(axis=0, how='all')
	if len(df)<1 :
		return None
	if df.index.dtype_str[:8]=='datetime':
		df['pbdate']=[int(x) for x in df.index.strftime('%Y%m%d')]
	else:
		df['pbdate']=[datetime.datetime.strptime(x,"%Y-%m-%d").strftime("%Y%m%d") for x in df.index]
	df['name']=symbol
	df=df[['open','high','low','close','volume','adjusted','pbdate','name']]
	return df

def hdr_data_fred(symbol,df):
	if df is None:
		return df
	df.rename(columns={symbol:"close"},inplace=True)
	df=df.dropna(axis=0, how='all')
	if len(df)<1 :
		return None
	df["pbdate"]=map(int,df.index.strftime("%Y%m%d"))
	df["name"]=symbol
	df=df[["close","pbdate","name"]]
	return df

def get_data_iex(ticker,start=None,end=None,days=730,tsTF=True,debugTF=False):
	(start,end) = get_start_end(start,end,days=days)
	days = (start-end).days
	yr = int(days)/365.0
	ranged = '5y' if yr>2 else '2y' if yr>1 else '1y' if yr>0.2 else '1m' if yr>0.02 else '5d'
	data = get_daily_iex(ticker,ranged=ranged,tsTF=tsTF,debugTF=debugTF)
	return data

def get_daily_iex(ticker,ranged='1y',types='chart',tsTF=True,debugTF=False):
	import pytz
	nytime = datetime.datetime.now(pytz.timezone('America/New_York'))
	curdate = int(nytime.strftime('%Y%m%d'))
	colLst = ["open","high","low","close","volume","adjusted","pbdate","name"]
	try:
		from iex_types_batch import iex_types_batch
		optx={'types':types,'ranged':ranged,'debugTF':debugTF,'saveDB':False,'outTF':False}
		data, _ = iex_types_batch([ticker],optx=optx)
		df = data[colLst]
		if df['pbdate'].iloc[-1] != curdate and nytime.hour>9:
			optx={'types':'quote','ranged':ranged,'debugTF':debugTF,'saveDB':False,'outTF':False}
			dx, _ = iex_types_batch([ticker],optx=optx)
			if debugTF is True:
				pqint( "==Add today's quote:" , file=sys.stderr)
				pqint( dx[colLst], file=sys.stderr)
			if dx['pbdate'].iloc[-1] > df['pbdate'].iloc[-1]:
				df = pd.concat([df,dx[colLst]])
		return df
	except Exception as e:
		pqint( str(e), 'failed to pull iex data', file=sys.stderr)
		return None

def get_minute_iex(ticker,ranged='1d',date=None,tsTF=True,debugTF=False):
	
	mdb = conn2mgdb(dbname='ara')
	tablename='iex_spark_hist'
	mCur=mdb[tablename]
	jobj = dict(ticker=ticker)
	dsp={x:1 for x in ['ticker','close','xclose','change','pchg','epochs','pbdt','pbdate','hhmm']}
	dsp.update(_id=0)
	mxdate=max(mCur.find(jobj).distinct("pbdate"))
	jobj.update(pbdate=mxdate)
	r = mCur.find(jobj,dsp,sort=[("epochs",-1)])
	df = pd.DataFrame(list(r))
	return df

def get_minute_iexOLD(ticker,ranged='1d',date=None,tsTF=True,debugTF=False):
	from iex_types_batch import iex_types_batch
	optx={'ranged':ranged,'debugTF':debugTF,'saveDB':False,'outTF':False,'tsTF':tsTF,'date':date}
	data, _ = iex_types_batch([ticker],optx=optx)
	if data.shape[0]<1:
		return None
	if 'marketVolume' in data:
		data['volume'] = data['marketVolume']
	return data[["open","high","low","close","volume","epochs","ticker"]]

def get_minute_yh(ticker,gap='1m',ranged='1d',tsTF=True,pchgTF=False,debugTF=False,**optx):
	from yh_hist_batch import yh_hist
	data = yh_hist(ticker,gap=gap,ranged=ranged,tsTF=tsTF,pchgTF=pchgTF,debugTF=debugTF)
	return data

def get_data_yh(ticker,start=None,end=None,days=1090,gap='1d',ranged=None,**optx):
	try:
		from yh_hist_batch import yh_hist
		if ranged is not None and len(ranged)>1:
			pass
		elif start is not None and end is not None:
			ranged = "{},{}".format(start,end)	
		elif start is not None:
			ranged = "{},".format(start)	
		elif end is not None:
			ranged = ",{}".format(end)	
		else:
			ranged = '{}d'.format(int(days/365.0*261+0.5))
		data = yh_hist(ticker,gap=gap,ranged=ranged,**optx)
		return data
	except Exception as e:
		pqint( str(e), 'failed to pull yh data', file=sys.stderr)
		return None

def get_datai(ticker,start=None,end=None,src='iex',days=730,debugTF=False,**optx):
	return pull_stock_data(ticker,start=start,end=end,src=src,days=days,debugTF=debugTF,**optx)

def get_dataiOLD(ticker,start=None,end=None,src="yahoo",days=1090,debugTF=False):
	""" get ticker history data from internet site (default: yahoo)
	"""
	import pandas_datareader.data as web
	data = None
	(start,end) = get_start_end(start,end,days=days)
	if debugTF:
		pqint( 'Pulling {0} from {1} to {2} in {3}'.format(ticker,start,end,src), file=sys.stderr)
	try:
		data = web.DataReader(ticker,src,start,end)
	except Exception as e:
		pqint( str(e), 'failed to pull pricing data', file=sys.stderr)
		return None
	try:
		fncname = hdr_data_fred if src=="fred" else hdr_data_stock
		data = fncname(ticker,data)
		if debugTF:
			pqint( data.iloc[[0,-2,-1]], file=sys.stderr)
	except Exception as e:
		pqint( str(e), 'failed to cleaning pricing data', file=sys.stderr)
	return data

def get_datax(ticker,start=None,end=None,dbname="ara",hostname="localhost",pgDB=None,colname='name',tablename='prc_hist',days=1090,debugTF=False):
	
	""" get stock history data from database
	"""
	global engine
	(start,end) = get_start_end(start,end,days=days)

	pbstart=int(start.strftime('%Y%m%d'))
	pbend=int(end.strftime('%Y%m%d'))
	dbURL='postgresql://sfdbo@{0}:5432/{1}'.format(hostname,dbname)
	if pgDB is not None:
		engine= pgDB
	else:
		engine= conn2db(dbURL=dbURL)
	xqr="SELECT open,high,low,close,volume,adjusted,pbdate,{3} FROM {4} WHERE {3}='{0}' and pbdate>={1} and pbdate<={2} ORDER BY pbdate".format(ticker,pbstart,pbend,colname,tablename)
	if debugTF:
		pqint( 'Selecting {0} from {1} to {2} in {3}'.format(ticker,pbstart,pbend,engine), file=sys.stderr)
		pqint( xqr, file=sys.stderr)
	try:
		data=pd.read_sql(xqr,con=engine)
		if debugTF:
			pqint(data.tail(2), file=sys.stderr)
	except Exception as e:
		pqint( str(e), 'failed to pull pricing data', file=sys.stderr)
		return None
	return data

def get_globalmacrox(ticker,start=None,end=None,dbname="ara",hostname="localhost",pgDB=None,colname='series',tablename='macro_hist_fred',days=1090,debugTF=False):
	
	""" get globalmacro history data from database
	"""
	global engine
	(start,end) = get_start_end(start,end,days=days)

	pbstart=int(start.strftime('%Y%m%d'))
	pbend=int(end.strftime('%Y%m%d'))
	dbURL='postgresql://sfdbo@{0}:5432/{1}'.format(hostname,dbname)
	if pgDB is not None:
		engine= pgDB
	else:
		engine= conn2db(dbURL=dbURL)
	#xqTmp="SELECT series,value,pbdate FROM {4} WHERE {3}='{0}' and pbdate>={1} and pbdate<={2} ORDER BY pbdate"
	xqTmp="SELECT p.*,m.freq,m.label,m.label_cn FROM (SELECT * FROM {4} WHERE {3}='{0}' and pbdate>={1} and pbdate<={2}) as p LEFT JOIN mapping_series_label m ON p.series=m.series ORDER BY pbdate"
	xqr=xqTmp.format(ticker,pbstart,pbend,colname,tablename)
	if debugTF:
		pqint( 'Selecting {0} from {1} to {2} in {3}'.format(ticker,pbstart,pbend,engine), file=sys.stderr)
		pqint( xqr, file=sys.stderr)
	try:
		data=pd.read_sql(xqr,con=engine)
		data.rename(columns={"value":"close","series":"name"},inplace=True)
		if debugTF:
			pqint(data.tail(2), file=sys.stderr)
	except Exception as e:
		pqint( str(e), 'failed to pull fred data', file=sys.stderr)
		return None
	return data

def pullStockHistory(ticker, **opts):
	"""
	Description: Pull Stock History from database or internet 
	Return: Dataframe[open,high,low,close,volume,adjusted,pbdate,name]
	Parameters:
	  ticker: stock ticker 
	  option_list default values:
		start=None,end=None,dbname='ara',hostname='localhost',
		pgDB=None,src='yahoo',days=1090,debugTF=False
	"""
	#opts=subDict(option_list, ['src','end','pgDB','debugTF','hostname','days','start','dbname','searchDB'])
	return pull_stock_data(ticker,**opts)

def psd(ticker, **opts):
	''' Pull stock data in time-series 
	Example,
	psd('AAPL',ranged='3mo',gap='1d') # daily past 3-month
	psd('AAPL',ranged='20210701,',gap='1d') # daily since 2021/07/01
	psd('AAPL',ranged='1d',gap='5m') # every 5-minute intraday 
	'''
	return pull_stock_data(ticker,**opts)

@saferun
def pull_stock_data(ticker,start=None,end=None,dbname="ara",hostname="localhost",pgDB=None,src='yh',days=1090,searchDB=True,debugTF=False,pchgTF=False,gap='1d',ranged='1d',tsTF=True,**opts):
	"""
	Get stock/macro historic prices from database or internet source:[src]
	in dataframe with columns: open,high,low,close,volume,adjusted,pbdate,name
	Note,
	  1. directly search web if searchDB is False
	  2. gap=[1m, 5m] will directly search web minute data
	  3. when 'ranged' is used, it has a higher priority than [days,start,end]
	"""
	datax={}
	if gap[-1:]=='m' and src not in ['fred','deriv']:
		from yh_hist_batch import yh_hist
		datax = get_minute_yh(ticker,gap=gap,ranged=ranged,tsTF=tsTF,pchgTF=pchgTF,debugTF=debugTF)
		datax = {} if datax is None or len(datax)<1 else datax
		sys.stderr.write("---pull_stock_data from minute[{}]:\n".format(len(datax)))
	elif searchDB is True:
		datax = data_from_db(ticker,start=start,end=end,dbname=dbname,hostname=hostname,pgDB=pgDB,src=src,days=days,debugTF=debugTF)
		datax = {} if datax is None or len(datax)<1 else datax
		sys.stderr.write("---pull_from_db from DB[{}]:\n".format(len(datax)))
	sys.stderr.write("---pull_stock_data:{}\n".format(1))
	if len(datax)<1 and ticker[-7:]!='_PCTCHG':
		datax = data_from_web(ticker,start=start,end=end,src=src,days=days,debugTF=debugTF,pchgTF=pchgTF)
		datax = {} if datax is None or len(datax)<1 else datax
		sys.stderr.write("---pull_stock_data from WEB[{}]:\n".format(len(datax)))
	sys.stderr.write("---pull_stock_data:{}\n".format(2))
	try:
		if len(datax)>1 and 'close' in datax and pchgTF and 'pchg' not in datax:
			datax['pchg'] = datax['close'].pct_change()
	except Exception as e:
		pass
	if len(datax)>0 and isinstance(datax,pd.DataFrame):
		sys.stderr.write("{}:\n".format(datax.head(2)))
		sys.stderr.write("{}:\n".format(datax.tail(2)))
	else:
		sys.stderr.write("**WARNINGS:No {} data from {}:\n".format(ticker,src))
	if tsTF is False:
		datax.reset_index(drop=True,inplace=True)
	return datax

def data_from_web(ticker,start=None,end=None,src='iex',days=730,debugTF=False,**optx):
	if any([x in ticker for x in ['^','=']]) is True or src.lower() in ['yahoo','yh'] or ticker[-3:] in ['.HK','.SS','.SZ','.TW'] or ticker[-2:] in ['.T']:
		data=get_data_yh(ticker,start,end,days=days,**optx)
	elif src=="iex" and re.search(r'[=^]',ticker) is None:
		data=get_data_iex(ticker,start,end,days=days)
	else:
		data=get_datai(ticker,start,end,src=src,days=days,debugTF=debugTF)
	return data

def data_from_db(ticker,start=None,end=None,dbname="ara",hostname="localhost",pgDB=None,src='iex',days=1090,debugTF=False):
	fncname = get_globalmacrox if src in ["fred","deriv"] else get_datax
	if src=="iex" and re.search(r'[=^]',ticker) is None:
		#tablename="prc_hist_iex" 
		tablename="prc_hist" 
	elif src in ["fred","deriv"]:
		tablename="macro_hist_fred" 
	else:
		tablename="prc_hist"
	datax=fncname(ticker,start,end,dbname,hostname=hostname,pgDB=pgDB,days=days,tablename=tablename,debugTF=debugTF)
	if datax is None or len(datax) < 1 :
		return {}
	idxtm=map(lambda x:datetime.datetime.strptime(str(x),"%Y%m%d"),datax['pbdate'])
	datax.set_index(pd.DatetimeIndex(idxtm),inplace=True)
	return datax

def get_csvdata(tkLst,start=None,end=None,dbname="ara",hostname="localhost",pgDB=None,src='yahoo',days=730,sep='|',columns=None,searchDB=True,debugTF=False):
	"""
	Get stock data in datafram with selected [columns] 
	OR
	from database hostname::dbname 
	OR
	from internet source: [src]
	"""
	if isinstance(tkLst,pd.DataFrame):
		df = tkLst
		if columns is not None and df.size > 0:
			df =  df[ list(set(df.columns) & set(columns.split(','))) ]
		return df
	if len(tkLst)<1:
		return None
	filename=tkLst[0]
	if filename=='-':
		df=pd.read_csv(sys.stdin,sep=sep)
	elif src is not None:
		df = pullStockHistory(filename,days=days,src=src,start=start,end=end,dbname=dbname,hostname=hostname,pgDB=pgDB,searchDB=searchDB,deugTF=debugTF)
	else:
		df = pd.read_csv(filename,sep=sep)
	if df.size < 1:
		pqint( "**ERROR: Data not found!", file=sys.stderr)
		return {}
	if columns is not None:
		df =  df[ list(set(df.columns) & set(columns.split(','))) ]
	return df

def rsi_signal(r,lh=[30,70]):
	if(r is np.nan or r is None ): return(0)
	if(r<lh[0]): return(r-lh[0])
	if(r>lh[1]): return(r-lh[1])
	return(0)

def calc_gkhv(o,h,l,c):
	""" Garman-Klass (GK) volatility
	"""
	from math import log,sqrt
	s1=s2=0
	for j,(oj,hj,lj,cj) in enumerate(zip(o,h,l,c)):
		s1 += log(hj/lj)**2
		s2 += log(cj/oj)**2
	n = len(o)
	s1 = s1*0.5/n
	s2 = s2*(2*log(2)-1)/n
	gkhv = sqrt(s1-s2)
	return gkhv

#Relative Strength Index
def calc_rsi(prices, n=14, method='ewm'):
	'''
	Calc Relative Strength Index
	number tied to 'ta==0.3.8' if method='ewm' or 'ema'
	number tied to 'ta==0.7.0' and 'pandas_ta' if method='wma'
	number tied to 'talib' if method='rolling'
	'''
	deltas = np.diff(prices,prepend=prices[0])
	UpI = np.where(deltas>0, deltas,0)
	DoI = np.where(deltas<0,-deltas,0)
	UpI = pd.Series(UpI)
	DoI = pd.Series(DoI)
	if method.lower() in ['ewm','ema']:
		PosDI = pd.Series(pd.Series.ewm(UpI, span = n, min_periods = n-1).mean())
		NegDI = pd.Series(pd.Series.ewm(DoI, span = n, min_periods = n-1).mean())
	elif method.lower()=='wma':
		PosDI = pd.Series(pd.Series.ewm(UpI, alpha = 1.0/n, min_periods = n-1, adjust=False).mean())
		NegDI = pd.Series(pd.Series.ewm(DoI, alpha = 1.0/n, min_periods = n-1, adjust=False).mean())
	else:
		PosDI = pd.Series(pd.Series.rolling(UpI, window = n, min_periods = n ).mean())
		NegDI = pd.Series(pd.Series.rolling(DoI, window = n, min_periods = n ).mean())
	#RSI = pd.Series(100 * PosDI / (PosDI + NegDI), name = 'RSI_' + str(n))
	#df = df.join(RSI)
	#return df
	RSI = 100 * PosDI / (PosDI + NegDI)
	return RSI

# DEPRECATD, inconsistent RSI formula
def calc_rsiOLD(prices, n=14):
	"""
	compute the n period relative strength indicator
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

def calc_macd(vd,nFast=12,nSlow=26,nSig=9,debugTF=False):
	''' calc RSI & MACD based on numpy vector vd
	return [rsi,emaSlow,emaFast,macdEma,macdCnt,macdCntCross]
	'''
	emaSlow = pd.Series(vd).ewm(span=nSlow).mean()
	emaFast = pd.Series(vd).ewm(span=nFast).mean()
	macdEma = emaFast-emaSlow
	macdCnt = macdEma.ewm(span=nSig).mean() #-- "signal_macd"
	macdCntCross = macdEma - macdCnt #-- "signal_value_macd"
	return [emaSlow,emaFast,macdEma,macdCnt,macdCntCross]

def run_macd(df,pcol='close',nFast=12,nSlow=26,nSig=9,nRsi=14,lh=[30,70],debugTF=False):
	""" MACD Signal Line & Centerline Crossovers return addtional series:
		rsi,rsi_sg,ema_slow,ema_fast,macd_ema,signal_macd,signal_value_macd,
		crossover_macd,crossover_centerline,signal_buysell_macd
	where
		rsi: RSI 
		rsi_sg: singal based on 'rsi' and high-low band 'lh'
		ema_slow: macd slow m.a. series default to nSlow=26
		ema_fast: macd fast m.a. series default to nFast=12
		macd_ema: macd_fast - ema_slow
		signal_macd: macd_ema m.a. series default to nSig=9
		signal_value_macd: macd_ema - signal_macd
		crossover_macd: sign of macd_ema
		crossover_centerline: sign of signal_value_macd
		signal_buysell_macd: 2 * sign of (difference of crossover_centerline)
	cf.
		https://www.linkedin.com/pulse/python-tutorial-macd-signal-line-centerline-andrew-hamlet
	"""
	vd = df[pcol].values

	##-- calc RSI
	rsi = calc_rsi(vd,nRsi)

	##-- calc MACD
	emaSlow,emaFast,macdEma,macdCnt,macdCntCross = calc_macd(vd,nFast=nFast,nSlow=nSlow,nSig=nSig,debugTF=debugTF)

	##-- calc RSI & MACD relevant signals
	rsiSg=np.vectorize(rsi_signal,excluded=['lh'])(rsi,lh=lh)
	macdEmaSg = np.sign(macdEma).astype(int) #-- "crossover_macd"
	macdCntCrossSg = np.sign(macdCntCross).astype(int) #-- "crossover_centerline"
	macdBuySell = 2*np.sign((macdCntCrossSg - macdCntCrossSg.shift(1))) #-- "signal_buysell_macd"

	##-- assign vectors to dataframe
	df['price']=df[pcol]
	colx= ["rsi","rsi_sg","ema_slow","ema_fast","macd_ema","signal_macd","signal_value_macd","crossover_macd","crossover_centerline","signal_buysell_macd"]
	coly=[rsi,rsiSg,emaSlow,emaFast,macdEma,macdCnt,macdCntCross,macdEmaSg,macdCntCrossSg,macdBuySell]
	for ky,val in zip(colx,coly):
		df[ky] = val
	return df

# DEPRECATED, use calc_macd() instead
def run_macd_OLD(df,pcol='close',nFast=12,nSlow=26,nSig=9,debugTF=False):
	""" MACD Signal Line & Centerline Crossovers
		https://www.linkedin.com/pulse/python-tutorial-macd-signal-line-centerline-andrew-hamlet
		DEPRECATED, use calc_macd() instead
	"""
	df['price']=df[pcol]
	df['rsi']=calc_rsi(df['price'])
	df['rsi_sg'] = df['rsi'].apply(rsi_signal)
	#df['ema_slow'] = pd.ewma(df['price'], span=nSlow) 
	#df['ema_fast'] = pd.ewma(df['price'], span=nFast) 
	df['ema_slow'] = df['price'].ewm(span=nSlow).mean()
	df['ema_fast'] = df['price'].ewm(span=nFast).mean()
	df['macd_ema'] = (df['ema_fast'] - df['ema_slow'])
	#df['signal_macd'] = pd.ewma(df['macd_ema'], span=nSig)
	df['signal_macd'] = df['macd_ema'].ewm(span=nSig).mean()
	df['signal_value_macd'] = df['macd_ema'] - df['signal_macd']
	df['crossover_macd'] = np.where(df['macd_ema'] > 0, 1, 0)
	df['crossover_macd'] = np.where(df['macd_ema'] < 0, -1, df['crossover_macd'])
	df['crossover_centerline'] = np.where(df['macd_ema'] > df['signal_macd'], 1, 0)
	df['crossover_centerline'] = np.where(df['macd_ema'] < df['signal_macd'], -1, df['crossover_centerline'])
	df['signal_buysell_macd'] = (2*(np.sign(df['crossover_centerline'] - df['crossover_centerline'].shift(1))))
	if debugTF:
		sys.stderr.write("{}\n{}\n".format("===MACD:",df.tail()))
	return df

def run_ohlc(dx,pcol='close',debugTF=False):
	""" calc OLHC relevant differences and ratios such as
		selling_ratio: open-low over high-low gap as selling pressure ratio, range [0,1]
		buying_ratio: high-open over high-low gap as buying force ratio, range [0,1]
		candle_ratio: close-open over high-low gap, range [-1,1]
	"""
	if 'open' not in dx:
		for clx in ['hiop','oplo','hilo','clop','selling_ratio','buying_ratio','selling_pressure','candle_ratio']:
			dx.loc[:,clx]=0.0
		for clx in ['close','open','high','low']:
			if clx not in dx:
				dx[clx] = dx[pcol].values
		if debugTF:
			sys.stderr.write("{}\n{}\n".format("===OHLC:",dx.tail()))
		return dx
	dx['hiop']= dx['high']-dx['open'] # buy force
	dx['oplo']= dx['open']-dx['low'] # sell force
	dx['hilo']= dx['high']-dx['low'] # high-low range, essential number
	dx['clop']= dx['close']-dx['open'] # daily force, essential number
	dx['selling_ratio']= dx['oplo']/dx['hilo'] # selling force ratio, 1-selling_ratio
	dx['buying_ratio']= dx['hiop']/dx['hilo'] # buying force ratio, essential [0,1]
	dx['selling_pressure']= -dx['selling_ratio'] # selling pressure ratio
	dx['candle_ratio']= dx['clop']/dx['hilo'] # candle ratio, essential [-1,1]
	if debugTF:
		sys.stderr.write("{}\n{}\n".format("===OHLC:",dx.tail()))
	return dx

def calc_ma_diff(px,window=5,debugTF=False):
	""" calc MA(5) of pandas.Series 'px' and difference btw 'px' and  MA(5)
	"""
	vu=px.rolling(window=window).mean()
	du=px-vu
	return [vu,du]

def run_sma(dx,pcol="close",winLst=[5,10,20],debugTF=False):
	""" run SMA and merge to dataframe:[dx]
	"""
	px=dx[pcol]
	for j,xn in enumerate(winLst):
		vu,du=calc_ma_diff(px,window=xn,debugTF=debugTF)
		vname,dname = ( 'ma{}'.format(xn),'dxma{}'.format(xn) )
		dx[vname]=vu
		dx[dname]=du
	return dx

def run_tech(dx,pcol='close',winLst=[5,10,20],macdSpan=[12,26,9,14],debugTF=False,nanTF=False,tsTF=False,copyTF=False,ohlcTF=True):
	""" run OLHC, SMA, MACD and return a dataframe 'dm'
	where
		pcol is selected column string in dataframe 'dx' for macd calculation
		winLst is for calc_sma, moving-average window-spans  default to [5,10,20]
		macdSpan is for calc_macd, window-spans for nFast,nSlow,nSig,nRsi default to [12,26,9,14]
	"""
	dm = dx.copy() if copyTF else dx

	# calc OLHC
	if ohlcTF:
		dm=run_ohlc(dm,pcol=pcol,debugTF=debugTF)

	# calc SMA
	run_sma(dm,pcol=pcol,winLst=winLst,debugTF=debugTF)

	# calc rsi signal
	nFast,nSlow,nSig,nRsi = macdSpan
	run_macd(dm,pcol=pcol,nFast=nFast,nSlow=nSlow,nSig=nSig,debugTF=debugTF)

	# keep NaN rows if nanTF is True
	if nanTF is False:
		dm=dm.dropna(axis=0, how='any')

	# set time-series index
	if tsTF:
		from _alan_date import ymd2dt
		tsname= 'epochs' if 'epochs' in dm else 'pbdate'
		if tsname in dm:
			idxtm=ymd2dt(dm[tsname].values)
			dm.set_index(pd.DatetimeIndex(idxtm),inplace=True)
	if debugTF:
		sys.stderr.write("{}\n{}\n".format("===run_tech:",dm.head()))
		sys.stderr.write("{}\n".format(dm.tail()))
	return dm

def batch_alan_calc(tkLst,dbname='ara',ohlc_pnl='ohlc_pnl',src='yh',
	tablename='ohlc_hist',sep='|',wmode='fail',debugTF=False,
	output='csv',hostname='localhost',end=None,
	days=730,smaWin='5,10,20',start=None,searchDB=True,saveDB=True,
	indexTF=False,ohlc_pattern='ohlc_pattern',outfile='-',**optx):
	"""
	Run OLHC, SMA, MACD of tkLst
	return: dataframe of OLHC, SMA, MACD of tkLst 
	"""
	opts=locals()
	opts.pop('tkLst',None)
	opts.pop('optx',None)
	if len(optx)>0:
		opts.update(optx)
	#for ky,va in opts.items():
	#	exec("{}=va".format(ky))

	winLst = [int(x) for x in smaWin.split(',')]
	pgDB=conn2pgdb(dbname=dbname,hostname=hostname)
	ret=None
	for j,ticker in enumerate(tkLst):
		try:
			dx=pullStockHistory(ticker,**opts)
		except:
			pqint( "***DATA ERROR @ batch_alan_calc():", sys.exc_info()[1], file=sys.stderr)
			continue
		if dx is None or len(dx)<1 :
			continue
		ret=run_tech(dx,pcol='close',winLst=winLst,debugTF=debugTF,nanTF=True)
		ret['ticker']=ticker
		if tablename is None or len(tablename)<1 or wmode not in ['replace','append'] :
			fp = sys.stdout if outfile in ['-','stdout'] else sys.stderr if outfile =='stderr' else open(outfile,'w')
			if output == 'dict' :
				fp.write( json.dumps(ret.to_dict()) )
			elif output == 'json' :
				fp.write( json.dumps(ret.to_json(orient='records')) )
			else :
				fp.write( ret.to_csv(sep=sep,index=indexTF) )
		else:
			ret.to_sql(tablename, pgDB, schema='public', index=True, if_exists=wmode)
			pqint( ret.tail(2).to_csv(sep="\t"), file=sys.stderr)
		j=j+1
	return ret

def ewma(x,span=None): return pd.Series(x).ewm(span=span).mean()

def sma(x,span=1): return pd.Series(x).rolling(window=span).mean()

def hw2ewma(x,span=1,beta=0 ):
	""" calc Holt-Winters 2nd-order EWMA
	"""
	N = x.size
	alpha = 2.0 / ( 1 + span )
	s = np.zeros(( N, ))
	b = np.zeros(( N, ))
	s[0] = x[0]
	for i in range( 1, N ):
		s[i] = alpha * x[i] + ( 1 - alpha )*( s[i-1] + b[i-1] )
		b[i] = beta * ( s[i] - s[i-1] ) + ( 1 - beta ) * b[i-1]
	return s
 
def ewma_smooth(x,span=5,direction='all'):
	ct = 0
	if direction.lower() != 'forward':
		fwd = ewma( x, span=span ) # take EWMA in fwd direction
		ct +=1
	if direction.lower() != 'backward':
		bwd = ewma( x[::-1], span=span ) # take EWMA in bwd direction
		ct +=10
	if ct < 11:
		return fwd if ct < 10 else bwd
	c = np.vstack(( fwd, bwd[::-1] )) # lump fwd and bwd together
	m = np.mean( c, axis=0 ) # average
	return m

def find_mnmx(va):
	''' find the indices of start/min/max/end of 1-dimensional ndarray
	'''
	ns=va.size
	vb=va[::-1]
	imn=ns-np.nanargmin(vb)-1
	imx=ns-np.nanargmax(vb)-1
	return [0,imn,imx,ns-1]

def mnmx_spchg(ranged,pchg,pbdate):
	"""
	support function for find_mnmx_wmqy()
	calc minx/max w.r.t. range [W,M,Y,YTD]
	"""
	mn, mx, ns = pchg.idxmin(),pchg.idxmax(),pchg.size
	return [ranged, pbdate.loc[mn],pchg.loc[mn],pbdate.loc[mx],pchg.loc[mx],ns]

def find_mnmx_wmqy(ddf,pcol='close',chgcol='pchg'):
	"""
	calc minx/max change in percentage change 
	in range period [W,M,Y,YTD] as month,quarter,year-to-date
	"""
	from _alan_date import s2dt,next_date as nd
	cdate = int(ddf['pbdate'].iloc[-1])
	pkys = ['range','mndate','mnpchg','mxdate','mxpchg','ns']
	mnmxV =[]
	if pcol!=chgcol:
		ddf[chgcol] = ddf[pcol].pct_change()
	# min/max of since the Friday of the week before
	ytdate=int(nd(cdate,days=-13,weekday=4,dformat='%Y%m%d',dtTF=False))
	spchg = ddf[ddf.pbdate>ytdate][chgcol]
	#spchg = ddf[chgcol].iloc[-5:]
	ret = mnmx_spchg('W',spchg,ddf.pbdate)
	mnmxV.append( dict(zip(pkys,ret)) )

	# min/max of the month
	ytdate=int(nd(cdate,day=1,dformat='%Y%m%d',dtTF=False))
	spchg = ddf[ddf.pbdate>=ytdate][chgcol]
	#spchg = ddf[chgcol].iloc[-22:]
	mn1m, mx1m, n1m = spchg.idxmin(),spchg.idxmax(),spchg.size
	ret = mnmx_spchg('M',spchg,ddf.pbdate)
	mnmxV.append( dict(zip(pkys,ret)) )

	# min/max of the quarter
	mon = s2dt(cdate).month
	qmon = int(round((mon+.5)/3)-1)*3+1
	ytdate=int(nd(cdate,month=qmon,day=1,dformat='%Y%m%d',dtTF=False))
	spchg = ddf[ddf.pbdate>=ytdate][chgcol]
	#spchg = ddf[chgcol].iloc[-65:]
	ret = mnmx_spchg('Q',spchg,ddf.pbdate)
	mnmxV.append( dict(zip(pkys,ret)) )

	# min/max since the beginning of year to date 
	#ytdate = int(ddf['pbdate'].iloc[-1])/10000*10000;
	ytdate=int(nd(cdate,month=1,day=1,dformat='%Y%m%d',dtTF=False))
	spchg = ddf[ddf.pbdate>=ytdate][chgcol]
	ret = mnmx_spchg('YTD',spchg,ddf.pbdate)
	mnmxV.append( dict(zip(pkys,ret)) )

	"""
	# min/max in a year
	spchg = ddf[chgcol]
	ret = mnmx_spchg('Y',spchg,ddf.pbdate)
	mnmxV.append( dict(zip(pkys,ret)) )
	"""
	return pd.DataFrame(data=mnmxV,columns=pkys).sort_values(by=['ns'],ascending=False)

def chk_mnmx(xchg,df):
	"""
	check if last xchg is another record date of min/max in range periods
	"""
	for xr in df.to_dict(orient='records'):
		if xchg>0 and xchg >= xr['mxpchg']:
			return [xr['range']+'_U',xr['mxdate'],xchg, xr['mxpchg'] ]
		elif xchg<0 and xchg <= xr['mnpchg']:
			return [xr['range']+'_D',xr['mndate'],xchg, xr['mnpchg'] ]
	return ['', None, xchg, None]

def list2chunk(v,n=100):
	''' partition an array 'v' into arrays limit to 'n' elements
	'''
	import numpy as np
	return [v[i:i+max(1,n)] for i in np.arange(0, len(v), n)]

def opt_alan_calc(argv=[],retParser=False):
	""" command-line options initial setup
	Arguments:
	argv:	list arguments, usually passed from sys.argv
	retParser:	OptionParser class return flag, default to False
	Return: (options, args) tuple if retParser is False else OptionParser class 
	"""
	parser = OptionParser(usage="usage: %prog [option] SYMBOL1 ...", version="%prog 0.71",
		description="ALAN basic calculation utility functions")
	parser.add_option("","--src",action="store",dest="src",default="yh",
		help="source [fred|yahoo|iex](default: yh)")
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
	parser.add_option("-t","--table",action="store",dest="tablename",default="ohlc_hist",
		help="db tablename for ohlc history (default: ohlc_hist)")
	parser.add_option("","--ohlc_pattern",action="store",dest="ohlc_pattern",default="ohlc_pattern",
		help="db tablename for strategy pattern (default: ohlc_pattern)")
	parser.add_option("","--ohlc_pnl",action="store",dest="ohlc_pnl",default="ohlc_pnl",
		help="db tablename for strategy pnl (default: ohlc_pnl)")
	parser.add_option("-w","--wmode",action="store",dest="wmode",default="replace",
		help="db table write-mode [replace|append|fail] (default: replace)")
	parser.add_option("","--no_database_save",action="store_false",dest="saveDB",default=True,
		help="no save to database (default: save to database)")
	parser.add_option("","--no_database_search",action="store_false",dest="searchDB",default=True,
		help="no database search (default: search database 1st before web search)")
	parser.add_option("","--show_index",action="store_true",dest="indexTF",default=False,
		help="show index (default: False) Note, OUTPUT ONLY")
	parser.add_option("","--outfile",action="store",dest="outfile",default="-",
		help="output filename (default: - as stdout)")
	parser.add_option("-o","--output",action="store",dest="output",
		help="output type (default: None)")
	parser.add_option("","--sep",action="store",dest="sep",default="|",
		help="outfile field separator (default: |)")
	parser.add_option("","--sma",action="store",dest="smaWin",default="5,10,20",
		help="SMA Windows (default: [5,10,20] days)")
	parser.add_option("","--extra_js",action="store",dest="extraJS",
		help="extra JSON in DICT format.")
	parser.add_option("","--extra_xs",action="store",dest="extraXS",
		help="extra excutable string in k1=v1;k2=v2; format")
	parser.add_option("","--debug",action="store_true",dest="debugTF",default=False,
		help="debugging (default: False)")
	(options, args) = parser.parse_args(argv)
	if retParser is True:
		return parser
	opts = vars(options)
	try:
		from _alan_str import extra_opts
		extra_opts(opts,xkey='extraJS',method='JS',updTF=True)
		extra_opts(opts,xkey='extraXS',method='XS',updTF=True)
	except Exception as e:
		pqint( str(e), file=sys.stderr)
	if 'sep' in opts:
		opts['sep']=opts['sep'].encode().decode('unicode_escape') if sys.version_info.major==3 else opts['sep'].decode('string_escape')

	return (opts, args)

if __name__ == '__main__':
	""" run OLHC, SMA, MACD of tkLst
		Usage of:
		printf "TSM\nIBM\n" | _alan_calc.py
	"""
	(opts, args)=opt_alan_calc(sys.argv[1:])
	if len(args) == 0:
		pqint("\nRead from pipe\n\n", file=sys.stderr)
		tkLst = sys.stdin.read().strip().split("\n")
	else:
		tkLst = args
		if opts['output'] is None:
			opts['output'] = "csv"
		opts['wmode'] = "fail"
	batch_alan_calc(tkLst,**opts)
	exit(0)
