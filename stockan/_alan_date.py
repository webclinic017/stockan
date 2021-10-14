#!/usr/bin/env python 
"""
Program: _alan_date.py
Description:
	Date functions for ALAN,
	convert daily price into monthly and update to prc_m_hist
Functions:
	def s2dt(s=None,dformat=''):
	def get_start_end(start=None,end=None,dformat='',**kwargs):
	def str2epoch(s=None,dformat=''):
	def next_date(d=None,dformat='',dtTF=True,**kwargs):
	def next_month_date(d=None,months=1):
	def delta2dates(end,start,dformat="%Y%m%d",fq="D",rounding=0):
	def delta2datesOLD(end,start,dformat="%Y%m%d",fq="D",rounding=0):
	def ymd_diff(start,end,dformat="%Y%m%d"):
	def epoch_parser(x,s=1000): return datetime.fromtimestamp(int(x/s))
	def ymd_parser(x,fmt='%Y%m%d'): return datetime.strptime(str(x),fmt)
	def ymd2dt(s,dformat="%Y%m%d"):
	def ymd_delta(ymd=None,days=0,dformat="%Y%m%d"):
	def ymd_delta_OLD(ymd=None,days=0,dformat="%Y%m%d"):
	def dt2ymd(dt,dformat="%Y%m%d"):
	def ymd2dt_example():
	def dt2ymd_example():
	def freq_d2m(s,fq='M',method='last'):
	def daily2month(fromSql=None,tablename=None,dbname="ara",hostname="localhost",wmode='fail',fq='M',method='last'):
	def tg_next2week(pdt,cdt=None,nwk=1):
	def tg_latest2week(pdt,cdt=None,nwk=1):
Last mod., Tue Apr 30 10:32:36 EDT 2019
Version: 0.70
"""
import sys
import numpy as np
from math import isinf, log
from datetime import datetime,timedelta,date
from dateutil import relativedelta
from _alan_calc import subDict

def seOfDay(d, startOfDay=False, endOfDay=False):
	"""
	return datetime of the start/end of a day on the date 'd'
	"""
	if startOfDay is True:
		d = datetime.strptime(d.strftime('%Y%m%d'),'%Y%m%d')
	elif endOfDay is True:
		d = datetime.strptime(d.strftime('%Y%m%d'),'%Y%m%d') + timedelta(days=1) -  timedelta(seconds=1)
	return d

def s2dt(s=None,dformat='',startOfDay=False,endOfDay=False):
	"""
	convert any of [Ymd,Y-m-d,Y/m/d,epoch,epochUnix] to datetime
	or any string with date format 'dformat`
	return None if s is not a string or None
	"""
	if isinstance(s,date):
		return seOfDay(s,startOfDay=startOfDay,endOfDay=endOfDay)
	#elif any([True for x in (None,[],{},(),0,range(0),'') if x==s]):
	#elif s is None or (hasattr(s,'__len__') and len(s)<1):
	elif not s:
		d = datetime.now()
		return seOfDay(d,startOfDay=startOfDay,endOfDay=endOfDay)
	elif isinstance(s,float) and not isinf(s):
		s = str(int(s))
	elif isinstance(s,(int,np.integer)):
		s = str(s)
	elif not isinstance(s,str):
		return None
	if isinstance(dformat,str) and len(dformat)>1:
		d = datetime.strptime(s,dformat)
	elif len(s)>=10 and s.isdigit():
		d = datetime.fromtimestamp(int(s[:10]))
	elif len(s)==8 and s.isdigit():
		d = datetime.strptime(s,"%Y%m%d")
	elif len(s)==10 and '-' in s:
		d = datetime.strptime(s,"%Y-%m-%d")
	elif len(s)==10 and '/' in s:
		d = datetime.strptime(s,"%Y/%m/%d")
	else:
		d = datetime.now()
	return seOfDay(d,startOfDay=startOfDay,endOfDay=endOfDay)

def get_start_end(start=None,end=None,dformat='',startOfDay=False,endOfDay=False,**kwargs):
	"""
	return end and start dates, based on 'end', 'start' strings
	where
	'start' and 'end' can be any of [Ymd,Y-m-d,Y/m/d,epoch,epochUnix,None]
	Default to current datetime if 'end' is None,
	Default to 'end' - 'kwargs' options if 'start' is None 
	and only 'end' is applied if 'kwargs' is empty. 
	"""
	end=s2dt(end,dformat,startOfDay=startOfDay,endOfDay=endOfDay)
	if start is None:
		start = end	
		if kwargs:
			try:
				start -= relativedelta.relativedelta(**kwargs) 
			except Exception as e:
				sys.stderr.write("**ERROR regs relativedelta: {}\n".format(str(e)))
	else:
		start=s2dt(start,dformat,startOfDay=startOfDay,endOfDay=endOfDay)
	return (start,end)

def str2epoch(s=None,dformat='',startOfDay=False,endOfDay=False):
	"""
	return string of [Ymd,Y-m-d,Y/m/d,epoch,epochUnix] to epoch in integer
	"""
	e=s2dt(s,dformat,startOfDay=startOfDay,endOfDay=endOfDay)
	return e if e is None else int(e.strftime("%s"))

def next_date(d=None,dformat='',dtTF=True,startOfDay=False,endOfDay=False,**kwargs):
	""" return datetime w.r.t. datetime 'd' based on options 'kwargs'
	Where
	kwargs can be years, months, weeks, days, hours, minutes, seconds, microseconds
	as the delta difference from current 'd'
	and month (of the year), weekday (forthcoming, Monday:0), day (of the month)
	e.g.,
	# return 1st date of the last month from current datetime
	next_date(None,months=-1,day=1)
	# return 1st date of March from current datetime (with same year)
	next_date(None,month=3)
	# return last month from current datetime (with same year and day)
	next_date(None,dformat='%Y%m%d',months=-1,dtTF=False)
	# return last date of Feburary from current datetime (with same year)
	next_date(None,month=2,day=31)
	"""
	dt = s2dt(d,dformat)
	if len(kwargs)>0:
		dt = dt + relativedelta.relativedelta(**kwargs)
	dt = s2dt(dt,startOfDay=startOfDay,endOfDay=endOfDay)
	if len(dformat)>1 and dtTF is False:
		return dt.strftime(dformat)
	else:
		return dt

def next_month_date(d=None,months=1):
	""" return same datetime [d] for the next month """
	if not d:
		d = datetime.now()
	return d + relativedelta.relativedelta(months=months)

def delta2dates(end,start,dformat="%Y%m%d",fq="D",rounding=0):
	""" return difference of 2 dates, end - start in day, month or year
	""" 
	if fq.upper()=="M":
		md=relativedelta.relativedelta(ymd2dt(end,dformat),ymd2dt(start,dformat))
		xdif=md.years*12+md.months
		if rounding > 0:
			xsign = -1 if md.days < 0 else 1
			xdif=xdif+int((md.days+30*xsign)/31)
	elif fq.upper()=="Y":
		md=relativedelta.relativedelta(ymd2dt(end,dformat),ymd2dt(start,dformat))
		xdif=md.years
		if rounding == 1:
			xsign = -1 if md.days < 0 else 1
			xdif=xdif+int(md.months+int((md.days+30*xsign)/31) + 11*xsign)/12
	elif fq.upper()=="W":
		md=ymd2dt(end,dformat)-ymd2dt(start,dformat)
		if rounding == 1:
			xsign = -1 if md.days < 0 else 1
			xdif=int((md.days+6*xsign)/7)
		else:
			xdif=int(md.days/7)
	elif fq.upper()=="HOUR":
		md=ymd2dt(end,dformat)-ymd2dt(start,dformat)
		if rounding == 1:
			xsign = -1 if md.total_seconds() < 0 else 1
			xdif=int((md.total_seconds()+3540*xsign)/3600)
		else:
			xdif=int(md.total_seconds()/3600)
	elif fq.upper()=="MINUTE":
		md=ymd2dt(end,dformat)-ymd2dt(start,dformat)
		if rounding == 1:
			xsign = -1 if md.total_seconds() < 0 else 1
			xdif=int((md.total_seconds()+59*xsign)/60)
		else:
			xdif=int(md.total_seconds()/60)
	elif fq.upper()=="SECOND":
		md=ymd2dt(end,dformat)-ymd2dt(start,dformat)
		xdif=md.seconds
	else:
		md=ymd2dt(end,dformat)-ymd2dt(start,dformat)
		xdif=md.days
	return xdif

def delta2datesOLD(end,start,dformat="%Y%m%d",fq="D",rounding=0):
	""" return difference of 2 dates, end - start in day, month or year
	""" 
	if fq.upper()=="M":
		md=relativedelta.relativedelta(datetime.strptime(str(end),dformat),datetime.strptime(str(start),dformat))
		xdif=md.years*12+md.months
		if rounding == 1:
			xdif=xdif+int((md.days+30)/31)
	elif fq.upper()=="Y":
		md=relativedelta.relativedelta(datetime.strptime(str(end),dformat),datetime.strptime(str(start),dformat))
		xdif=md.years
		if rounding == 1:
			xdif=xdif+int(md.months+int((md.days+30)/31) + 11)/12
	elif fq.upper()=="W":
		md=datetime.strptime(str(end),dformat) - datetime.strptime(str(start),dformat)
		if rounding == 1:
			xdif=int((md.days+6)/7)
		else:
			xdif=int(md.days/7)
	else:
		md=datetime.strptime(str(end),dformat) - datetime.strptime(str(start),dformat)
		xdif=md.days
	return xdif

def ymd_diff(start,end,dformat="%Y%m%d"):
	""" return difference in days of end-start
	"""
	xdif=datetime.strptime(str(end),dformat) - datetime.strptime(str(start),dformat)
	return xdif.days

def epoch_parser(x,s=1000): return datetime.fromtimestamp(int(x/s)) if x is not None else x

def ymd_parser(x,fmt='%Y%m%d'): return datetime.strptime(str(x),fmt) if x is not None else x

def ymd2dt(s,dformat="%Y%m%d"):
	"""
	convert [s] into a datetime struct format based on format: [dformat]
	note:
	  1. s can be a string or a list of string
	  2. s is treated as a epoch number if s is digit and s >= 10 digits

	"""
	if isinstance(s,(list,tuple,np.ndarray)):
		return [ymd2dt(x,dformat=dformat) for x in s]
	if s is None or isinstance(s,date):
		return s
	elif (isinstance(s,(np.integer,int,float)) or s.isdigit() ): 
		s = int(s) 
		if log(s,10) >= 12:
			s = s/1000
		if log(s,10) >= 9:
			return datetime.fromtimestamp(s)
	return datetime.strptime(str(s),dformat)

def ymd_delta(ymd=None,days=0,dformat="%Y%m%d"):
	""" return date of past [days] from [ymd] date in [dformat] format
	"""
	if ymd is not None and len(str(ymd))>4 and dformat is not None and len(dformat)>1:
		dt=datetime.strptime(str(ymd),dformat)
	elif isinstance(ymd,datetime):
		dt=ymd
	else:
		dt=datetime.now()
	if days != 0:
		dt=dt-timedelta(days=days)
	if dformat is None or len(dformat)<2:
		ret = dt
	else:
		ret = datetime.strftime(dt,dformat)
	return ret

def ymd_delta_OLD(ymd=None,days=0,dformat="%Y%m%d"):
	""" return date in format: [dformat] of [days] days from [ymd]
	"""
	if ymd is None:
		dt=datetime.now()-timedelta(days=days)
	else:
		dt=datetime.strptime(str(ymd),dformat)-timedelta(days=days)
	return datetime.strftime(dt,dformat)

def dt2ymd(dt,dformat="%Y%m%d"):
	"""
	Convert datetime 'dt' to struct format 'dformat' 
	Note that s can be a string or a list of string
	"""
	if isinstance(dt,(list,tuple,np.ndarray)):
		return [dt2ymd(x,dformat) for x in dt]
	if isinstance(dt,date):
		return dt.strftime(dformat)
	elif not dt:
		return datetime.strftime(datetime.now(),dformat)
	else:
		return dt

def ymd2dt_example():
	ymd=dt2ymd(datetime.today())
	return 'ymd2dt({})={}'.format(ymd,ymd2dt(ymd))

def dt2ymd_example():
	return 'dt2ymd(today)={}'.format(dt2ymd(datetime.now()))

def freq_d2m(s,fq='M',method='last',dtcol='pbdate',debugTF=False):
	'''
	Convert pandas.Series 's' to new frequency 'fq' of method 'method'
	Note dtcol is used if s.index is not DatetimeIndex
	
	'''
	import pandas as pd
	freq = fq[:1].upper()
	mth = method.lower()
	if not isinstance(s,pd.DatetimeIndex):
		s.index= [ ymd2dt(x) for x in s[dtcol] ]
	try:
		ml = getattr(s.resample(freq),mth)()
	except Exception as e:
		sys.stderr.write("**ERROR:{}:{}\n".format(mth,e))
	if debugTF:
		sys.stderr.write("Method: {}\n".format(mth))
		sys.stderr.write("{}\n".format(ml.tail()))
	return ml

def daily2month(fromSql=None,tablename=None,dbname="ara",hostname="localhost",wmode='fail',fq='M',method='last',output=''):
	""" Get price history using yahoo and save them into 'prc_temp_yh'
	"""
	from sqlalchemy import create_engine
	import pandas as pd
	if not fromSql: 
		fromSql="SELECT * FROM prc_hist WHERE name='IBM' ORDER BY pbdate"
	if tablename is None:
		tablename="spdr_price_m_hist"
	# Connect to DB
	dbURL='postgresql://sfdbo@{0}:5432/{1}'.format(hostname,dbname)
	engine = create_engine(dbURL)
	# Get stock price history
	sDu = pd.read_sql(fromSql,con=engine) 

	# Get stock frequency conversion
	tsp = freq_d2m(sDu,method=method,fq=fq)

	# Save stock price history to {tablename}
	if wmode in ['replace','append'] :
		tsp.to_sql(tablename,engine,schema='public',index=False,if_exists=wmode)
	if output=='csv':
		sys.stdout.write(tsp.to_csv(index=False,sep="\t")+"\n")
	# Close DB
	engine.dispose()
	return tsp

def tg_next2week(pdt,cdt=None,nwk=1):
	"""
	check of target date 'pdt' is within up-to-next week Friday range
	"""
	if cdt is None:
		cdt = next_date()
	if pdt<cdt and pdt.day!=cdt.day:
		return {}
	gap = nwk
	xdt = next_date(cdt,weeks=gap,weekday=4) # monday of the last week
	sundt = next_date(xdt,days=-5) # coming sunday from xdt(monday)
	if pdt>xdt: # discard any data bigger than Friday of the next week
		sys.stderr.write("Too far to report:{} > {}, disregard!\n".format(pdt,xdt))
		return {}
	currWeekTF= (pdt<sundt)
	todayTF= (pdt.day==cdt.day)
	d=dict(currWeekTF=currWeekTF,todayTF=todayTF,endDate=xdt,currDate=cdt,tgDate=pdt)
	d.update(tgWeekday=pdt.weekday()+1)
	return d

def tg_latest2week(pdt,cdt=None,nwk=1):
	"""
	check of target date 'pdt' is within last week's Monday range
	"""
	if cdt is None:
		cdt = next_date()
	if pdt>cdt and pdt.day!=cdt.day:
		return {}
	gap = -nwk if cdt.weekday()==0 else -(nwk+1)
	xdt = next_date(cdt,weeks=gap,weekday=0) # monday of the last week
	sundt = next_date(xdt,days=5) # coming sunday from xdt(monday)
	if pdt<=xdt: # discard any data less Monday of the week
		sys.stderr.write("Stale news:{} < {}, disregard!\n".format(pdt,xdt))
		return {}
	currWeekTF= (pdt>sundt)
	todayTF= (pdt.day==cdt.day)
	d=dict(currWeekTF=currWeekTF,todayTF=todayTF,startDate=xdt,currDate=cdt,tgDate=pdt)
	d.update(tgWeekday=pdt.weekday()+1)
	return d

if __name__ == '__main__':
	fromSql="SELECT * FROM spdr_price_hist WHERE ticker='{}' ORDER BY pbdate"
	ticker=sys.argv[1] if len(sys.argv)>1 else "IBM"
	fromSql=fromSql.format(ticker)

	sys.stderr.write("{}\n".format(daily2month(fromSql).tail()))
	print("20180221-20170320 in months:",delta2dates(20180221,20170320,fq="M"))
