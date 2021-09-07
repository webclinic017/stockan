#!/usr/bin/env python
""" 
pull yahoo calendar by event [economic|earnings|splits|ipo]
and save info to table: [earnings_yh], [economic_yh]...
OR pull yahoo earnings history for specific stock 'ticker'
and save info to table: [earnings_hist_yh] DEPRECATED

Usage of:
python macro_event_yh.py --type=earnings --day=20190801
OR
python macro_event_yh.py --ticker=aapl
OR
python macro_event_yh.py --type=ipo --day=20190612
OR
python macro_event_yh.py --type=splits --day=20190813
OR
printf "aapl\nibm"| python macro_event_yh.py --type=earnings --ticker=-

Ref_uri:
https://finance.yahoo.com/calendar/earnings?day=2019-01-24
OR
https://finance.yahoo.com/calendar/earnings?symbol=AAPL
OR for specific date
https://finance.yahoo.com/calendar/{event}?day={yyyy-mm-dd}
OR for specific ticker
https://finance.yahoo.com/calendar/earnings?symbol={ticker}

Also see: eps_nasdaq.py
Last mod., Tue Jan 29 14:08:11 EST 2019
"""
import sys
from optparse import OptionParser
import pandas as pd
from sqlalchemy import create_engine, MetaData
from datetime import datetime
import numpy as np
from _alan_str import upsert_mdb

def get_datestr(xdate):
	if xdate is None:
		return datetime.today().strftime('%Y-%m-%d')
	if isinstance(xdate,float):
		xdate = str(int(xdate))
	elif isinstance(xdate,(int,np.integer)):
		xdate = str(xdate)
	if xdate.isdigit() and len(xdate)==8:
		d = xdate
		xdate = '{}-{}-{}'.format(d[:4],d[4:6],d[-2:])
	elif '/' in xdate:
		xdate.replace('/','-')
	return xdate

def earnings_morningstar(xdate=None,xtype='earnings',uri=None,dbname='ara',tablename='earnings_morningstar',debugTF=False,**optx):
	""" Usage of:
	python -c "from macro_event_yh import earnings_morningstar as em;em(xdate=20190430)"
	"""
	xdate=get_datestr(xdate)
	#http://www.morningstar.com/earnings/Handler/GetEarningsCalendar.ashx?rptDate=2019-05-01&r=68265
	if uri is None:
		urx="http://www.morningstar.com/earnings/Handler/GetEarningsCalendar.ashx?rptDate={}"
		uri = urx.format(xdate)
	ret=pd.read_html(uri)
	df = ret[1]
	v =[str(x.split()[-1].encode('ascii','ignore')) for x in df[u'Company Name']]
	df['ticker'] = v
	df['pbdate'] = int(xdate.replace('-',''))
	zpk = {'ticker','pbdate'} if xtype == 'earnings' else {'event','pbdate'}
	clientM=None
	print( v)
	mobj,clientM,err_msg = upsert_mdb(df,clientM,dbname=dbname,tablename=tablename,zpk=zpk)
	return df

def get_calendar_info(xdate,xtype='economic',uri=None,classname=None,debugTF=False,**optx):
	xdate = get_datestr(xdate)
	if uri==None:
		uri="https://finance.yahoo.com/calendar/{}?day={}".format(xtype,xdate)
	if classname==None:
		classname="data-table W(100%) Bdcl(c) Pos(r) BdB Bdc($c-fuji-grey-c)"
	url=uri
	for j in [0,1,2,3,4,5]:
		try:
			if j>0:
				url = uri + '&offset={}00&size=100'.format(j)
			sys.stderr.write( "{}\n.....\n".format(url)+"\n" )
			xa=pd.read_html(url,index_col=0,header=0)
			if len(xa[0])<1:
				break
			if j>0:
				df = pd.concat([df,xa[0]])
			else:
				df=xa[0]
		except Exception as e:
			sys.stderr.write( '**ERROR: {}\n{}\n'.format(str(e),url)+"\n" )
			break
	df['pbdate'] = int(xdate.replace('-',''))
	if debugTF:
		sys.stderr.write( 'DF:\n{}'.format(df)+"\n" )
	sys.stderr.write( 'Total list {}'.format(len(df))+"\n" )
	return df

def search_splits(xdate,xtype='ipo',uri=None,classname=None,debugTF=False,**optx):
	df = get_calendar_info(xdate,xtype=xtype,uri=uri,classname=classname)
	if len(df)<1:
		return {}
	df.columns = [x.replace(' ','').replace('?','') for x in df.columns]
	df = df.reset_index()
	return df

def search_ipo(xdate,xtype='ipo',uri=None,classname=None,debugTF=False,**optx):
	df = get_calendar_info(xdate,xtype=xtype,uri=uri,classname=classname)
	if len(df)<1:
		return {}
	df = df.query("Currency=='USD' & Actions=='Priced'")
	if len(df)<1:
		return {}
	df.columns = [x.replace(' ','') for x in df.columns]
	#hdr=['Symbol','Company','Exchange','Date','Price Range','Price','Currency','Shares','Actions']
	#df.columns = hdr
	df = df.reset_index()
	return df

def search_economic(xdate,xtype='economic',uri=None,classname=None,debugTF=False,**optx):
	df = get_calendar_info(xdate,xtype=xtype,uri=uri,classname=classname)
	if len(df)<1:
		return {}
	df = df.query("Country=='US' & Actual!='-'")
	hdr=['event','Country','EventTime','ForPeriod','Actual','MarketExpectation','Previous','RevisedFrom','pbdate']
	df = df.reset_index()
	df.columns = hdr
	return df

def search_earnings(xdate,xtype='earnings',uri=None,classname=None,debugTF=False,**optx):
	df = get_calendar_info(xdate,xtype=xtype,uri=uri,classname=classname)
	if len(df)<1:
		return {}
	hdr=['ticker','Company', 'CallTime', 'estimatedEPS', 'actualEPS', 'EPSSurprisePCT','pbdate']
	df = df.reset_index()
	df.columns = hdr
	return df

def search_earnings_by_symbol(ticker,uri=None,xtype='earnings',debugTF=False,**optx):
	if uri is None:
		urx='https://finance.yahoo.com/calendar/earnings?symbol={}'
		uri = urx.format(ticker)
	sys.stderr.write( "Pulling\n{}\n...".format(uri)+"\n" )
	try:
		xf = pd.read_html(uri,index_col=False,header=0)
	except Exception as e:
		sys.stderr.write( '**ERROR: {}:{}  @{}()'.format(ticker,str(e),'search_earnings_by_symbol')+"\n" )
		return {};
	if len(xf)<1:
		return {};
	df = xf[0]
	if debugTF:
		sys.stderr.write('{}'.format(df)+"\n" )
		sys.stderr.write('{}'.format(df.columns)+"\n" )
	if isinstance(df['EPS Estimate'].iloc[0],str):
		df = df.drop(df[df['EPS Estimate']=='-'].index).reset_index(drop=True)
		df['estimatedEPS'] = [ float(x) if x!='-' else None for x in df['EPS Estimate'].values ]
	if 'Reported EPS' in df.columns and isinstance(df['Reported EPS'].iloc[0],str):
		df['EPS'] = [ float(x) if x!='-' else None for x in df['Reported EPS'].values ]
	df.rename(columns={'Symbol':'ticker'},inplace=True)
	if 'Earnings Date' in df.columns:
		edt = [ datetime.strptime(x,'%b %d, %Y, %I %p%Z') for x in df['Earnings Date'].values ]
		df['pbdate']  = [ int(x.strftime('%Y%m%d')) for x in edt ]
		df.loc[:,'Earnings Date']  = edt
	return df

def run_macro_event_yh(uri=None,xdate=None,saveDB=True,tablename=None,wmode='replace',xtype='earnings',clientM=None,hostname='localhost',dbname=None,ticker=None,debugTF=False,**optx):
	""" find macro event from yahoo
	"""
	pgDB = None
	if ticker is not None:
		urx='https://finance.yahoo.com/calendar/earnings?symbol={}'
		uri = urx.format(ticker)
		funcName = 'search_earnings_by_symbol'
		tablename = "{}_hist_yh".format(xtype)
	else:
		funcName = "search_{}".format(xtype)
		tablename = "{}_yh".format(xtype) if not tablename else tablename
	if funcName in globals():
		searchFunc = globals()[funcName]
		dfyc=searchFunc(xdate,uri=uri,xtype=xtype)
		if len(dfyc)<1:
			return {}
	else:
		return {}
	
	if any([saveDB is False]):
		sys.stderr.write('{}'.format(dfyc.to_csv(index=False,sep='|'))+"\n" )
	else:
		zpk = {'ticker','pbdate'} if xtype == 'earnings' else {'event','pbdate'}
		mobj,clientM,err_msg = upsert_mdb(dfyc,clientM,dbname=dbname,tablename=tablename,zpk=zpk)
		sys.stderr.write( "{}\n...\n{}\n saved to {}:{}".format(dfyc.head(1),dfyc.tail(1),clientM,tablename)+"\n" )
	if pgDB is not None:
		dfyc.to_sql(tablename,pgDB,index=False,schema='public',if_exists=wmode)
	return dfyc

def opt_macro_event_yh(argv,retParser=False):
	parser = OptionParser(usage="usage: %prog [option]", version="%prog 0.1",
		description="get up-to-date calendar events")
	parser.add_option("","--uri",action="store",dest="uri",
		help="uri (default: None)")
	parser.add_option("","--ticker",action="store",dest="ticker",
		help="ticker (default: None), for [earnings] type ONLY")
	parser.add_option("","--day",action="store",dest="xdate",
		help="YYYYMMDD (default: today)")
	parser.add_option("","--type",action="store",dest="xtype",default="earnings",
		help="calendar type [economic|earnings](default: earnings)")
	parser.add_option("-d","--database",action="store",dest="dbname",default="ara",
		help="database (default: ara)")
	parser.add_option("","--host",action="store",dest="hostname",default="localhost",
		help="db host (default: localhost)")
	parser.add_option("-t","--table",action="store",dest="tablename",
		help="db tablename")
	parser.add_option("-w","--wmode",action="store",dest="wmode",default="replace",
		help="db table write-mode [replace|append|fail] (default: replace)")
	parser.add_option("","--no_database_save",action="store_false",dest="saveDB",default=True,
		help="no save to database (default: save to database)")
	parser.add_option("","--debug",action="store_true",dest="debugTF",default=False,
		help="DEBUG flag (default: False)")
	(options, args) = parser.parse_args(argv[1:])
	if retParser is True:
		return parser
	return (vars(options), args)

if __name__ == '__main__':
	opt,args =opt_macro_event_yh(sys.argv)
	if opt['ticker'] is not None:
		tkLst=opt['ticker'].split(',')
		if tkLst[0]=='-':
			tkLst = sys.stdin.read().split()
		for ticker in tkLst:
			opt['ticker'] = ticker
			opt['xtype'] = 'earnings'
			try:
				run_macro_event_yh(**opt)
			except Exception as e:
				sys.stderr.write( '**ERROR: {} @ {}'.format(ticker,str(e))+"\n" )
	else:
		run_macro_event_yh(**opt)
