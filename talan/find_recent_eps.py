#!/usr/bin/env python3
'''
pull most recent ESP 
table required: MDB::earnings_yh,earnings_zacks 
for last season or latest one respectively
Usage of:
python -c "ticker='AAPL';from find_recent_eps import find_recent_eps as findEps;df=findEps(ticker);print(df)"
'''
from _alan_str import find_mdb, num_en2cn
from _alan_date import next_date,s2dt
from _alan_calc import sqlQuery
import datetime
import pandas as pd
import numpy as np
import sys
import math

def select_eps_hist(ticker='AAPL',pbdate=None,types='earnings',dbname='ara',**optx):
	debugTF=optx.pop('debugTF',False)
	if types.lower() in ['earnings','eps']:
		tablename='earnings_yh'
	elif types.lower() == 'stats':
		tablename='qS_keyStatistics'
		jobj = {"ticker":ticker}
		ret,_,errmsg = find_mdb(jobj,dbname=dbname,tablename=tablename)
		if len(ret)<1:
			return {}
		df = pd.DataFrame(ret)
		return df
	else:
		return {}
	pbdate = int(next_date(pbdate,dformat='%Y%m%d',dtTF=False,days=-2))
	jobj = {"pbdate":{'$lt':pbdate},"ticker":ticker}
	ret,_,errmsg = find_mdb(jobj,dbname=dbname,tablename=tablename,sortLst=['pbdate'],ascendingTF=False)
	df = pd.DataFrame(ret)
	if debugTF:
		sys.stderr.write("EPS {} of {}:\n".format(ticker,tablename))
		sys.stderr.write("{}\n".format(df))
	if 'actualEPS' not in df:
		return df
	df = df.dropna(subset=['actualEPS'])
	df = df[~df['actualEPS'].isin(['-'])]
	df['actualEPS']=df['actualEPS'].astype(float)
	return df

# DEPRECATED, USE yahoo number
def select_eps_histIEX(ticker='UNP',types='eps',nrow=1):
	""" select eps/revenue/profitMaring from PSQL tables
	"""
	if types=='eps':
		xqTmp="SELECT * from iex_earnings_hist where ticker='{}' order by pbdate DESC limit {}"
	elif types=='stats':
		xqTmp="SELECT * from iex_stats_hist where ticker='{}' order by pbdate DESC limit {}"
	else:
		xqTmp="SELECT * from iex_financials_hist where ticker='{}' and freq='Q' order by pbdate DESC limit {}"
	xqr=xqTmp.format(ticker,nrow)
	df = sqlQuery(xqr)
	if len(df)<1:
		return {}
	return df

def calc_pchg(*eps):
	""" 
	calc pctChange if denominator eps[1] is positive
	and return (Change Percent, True
	otherwise (change differrene, False)
	"""
	if eps[1]<=0:
		pchg=eps[0]*1.0-eps[1]
	else:
		pchg=(eps[0]*1.0/eps[1]-1)
	return (pchg, eps[1]>0)

def find_past_eps(ticker='UNP',cdt=None,debugTF=False):
	dd=dict(ticker=ticker)
	dh = select_eps_hist(ticker=ticker,pbdate=cdt,debugTF=debugTF)
	if len(dh)>1:
		if len(dh)>4:
			xeps = dh['actualEPS'].iloc[4]
			dd.update(eps_chgFreq='Y')
		else:
			xeps = dh['actualEPS'].iloc[1]
			dd.update(eps_chgFreq='Q')
		ceps = dh['actualEPS'].iloc[0]
		pchg,chgTF = calc_pchg(ceps,xeps)
		dd.update(dh.iloc[0].to_dict())
		dd.update(eps_pctChange=pchg,eps=ceps,prev_eps=xeps,eps_isPct=chgTF)
		dd.update(latest2weekTF=True)
		dd.update(past_eps = dh[['actualEPS', 'pbdate']].set_index('pbdate'))
		#sys.stderr.write("dh:\n{}\n".format(dh[['actualEPS', 'pbdate']]))
	dh = select_eps_hist(ticker=ticker,types='financials',nrow=2,debugTF=debugTF)
	if len(dh)>1:
		xeps = dh['totalRevenue'].iloc[1]
		ceps = dh['totalRevenue'].iloc[0]
		pchg,chgTF = calc_pchg(ceps,xeps)
		dd.update(revenue_pctChange=pchg,revenue=ceps,prev_revenue=xeps)
		ceps,xeps = (dh['operatingIncome']/dh['operatingRevenue']).values[:2]
		#ceps,xeps = (dh['profitMargin']).values[:2]
		if not math.isnan(ceps):
			pchg,chgTF = calc_pchg(ceps,xeps)
			dd.update(profitMargin_pctChange=pchg,profitMargin=ceps,prev_profitMargin=xeps,profitMargin_isPct=chgTF)
	return(dd)

def find_latest_eps(ticker='UNP',cdt=None,dbname='ara',debugTF=False):
	"""
	Find eps since monday of the last week
	"""
	if not cdt:
		cdt = next_date()
	elif not isinstance(cdt,datetime.date):
		cdt = next_date(cdt)
	gap = -1 if cdt.weekday()==0 else -2
	xdt = next_date(cdt,weeks=gap,weekday=0) # monday of the last week
	sundt = next_date(xdt,days=6) # coming sunday from xdt(monday)
	xdate=int(xdt.strftime('%Y%m%d'))
	findDct={"pbdate":{'$gte':xdate},"ticker":ticker}
	# use zacks for up-to-date EPS
	ret,_,errmsg = find_mdb(jobj=findDct,dbname=dbname,tablename='earnings_zacks',field=['ticker','Estimate','Reported','pbdate','Time'],sortLst=['pbdate'],ascendingTF=False)
	if len(ret)>0:
		dd = ret[0]
		dd['eps'] = float(dd['Reported'])
		pdt = s2dt(dd['pbdate'])
		if pdt<=xdt: # discard any data less Monday of the week 
			sys.stderr.write("Stale news:{} < {}, disregard!\n".format(pdt,xdt))
			return {}
		currWeekTF= (pdt>sundt)
		todayTF= (pdt.day==cdt.day)
		dd.update(weekday=pdt.weekday()+1,currWeekTF=currWeekTF,todayTF=todayTF)
		pbdate=dd['pbdate']
	else:
		return {}
	dh = select_eps_hist(ticker=ticker,pbdate=pbdate,debugTF=debugTF)
	if len(dh)>0:
		ceps = dd['eps']
		if dh.shape[0]>3:
			xeps = dh['actualEPS'].iloc[3] 
			dd.update(eps_chgFreq='Y')
		else:
			xeps = dh['actualEPS'].iloc[0] 
			dd.update(eps_Freq='Q')
		pchg,chgTF = calc_pchg(ceps,xeps)
		dd.update(eps_pctChange=pchg,prev_eps=xeps,eps_isPct=chgTF)
		dd.update(latest2weekTF=True)
		d1 = dh[['actualEPS', 'pbdate']].set_index('pbdate')
		d1.loc[pbdate] = {'actualEPS':ceps}
		dd.update(past_eps = d1.sort_index())
	dh = select_eps_hist(ticker=ticker,types='financials',debugTF=debugTF)
	if len(dh) and 'revenue' in dd:
		revenueStr = dd['revenue']
		revenue = num_en2cn( revenueStr,numTF=True)
		pchg = revenue/dh['totalRevenue'].iloc[0]-1
		dd.update(revenue_pctChange=pchg,prev_revenue=xeps)
		dd.update(revenue=revenue,revenueStr=revenueStr)
	return(dd)

def find_recent_eps(ticker='UNP',cdt=None,debugTF=False):
	df = find_latest_eps(ticker,cdt=cdt,debugTF=debugTF)
	if len(df)<1:
		sys.stderr.write("{}\n".format("Latest EPS not found!"))
		df = find_past_eps(ticker,cdt=cdt,debugTF=debugTF)
	if 'pbdate' in df:
		pbdate=int(df['pbdate'])
		df.update(pbdate=pbdate)
	return df

if __name__ == '__main__':
	ticker = sys.argv[1] if len(sys.argv)>1 else 'AMZN'
	cdt = sys.argv[2] if len(sys.argv)>2 else None
	dd =find_recent_eps(ticker,cdt=cdt,debugTF=True)
	#sys.stdout.write("{}\n".format(dd))
	# serializing datetime and bool and dataframe for json
	import json,datetime
	#dtCvt = lambda x: x.__str__() if isinstance(x, (bool,datetime.datetime)) else ''
	def dtCvt(obj): 
		if isinstance(obj, (bool,datetime.datetime)):
			o = obj.__str__()
		elif isinstance(obj, pd.DataFrame):
			o= obj.to_json(orient='records')
		else:
			o= ''
		return o
	sys.stdout.write("{}\n".format(json.dumps(dd,default=dtCvt)))

