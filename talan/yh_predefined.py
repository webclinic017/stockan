#!/usr/bin/env python
'''
Get predefined data via query2.finance.yahoo.com/v1/finance/screener
Usage of:
python yh_predefined.py [ day_gainers | day_losers | most_actives ]

RUN _actives in 'volume'
python yh_predefined.py --no_database_save

OR yh_predefined_query() call 
python -c "from yh_predefined import yh_predefined_query as ypq;print(ypq())"

OR bb_predefined() call most_actives in 'volume' and '>3% price change'
python -c "from yh_predefined import bb_predefined as ypq;df=ypq('most_actives',addiFilter=1);print(df)"

Last Mod., Fri Jul 26 23:34:16 EDT 2019
'''
import sys
import requests
import pandas as pd
from _alan_str import write2mdb,find_mdb
from _alan_date import s2dt
from _alan_calc import sqlQuery
from _alan_optparse import parse_opt, subDict

def bb_predefined(scrIds='most_actives',dbname='ara',saveDB=True,mappingTF=True,mmGap=30,addiFilter=1,debugTF=False,**optx):
	''' 
	return a ticker list based on the combo of
	yahoo 'scrIds' list and BB internal tracking tickers
	where
	  mappingTF: apply list on in the [mapping_ticker_cik] table
	  addiFilter: adds additional query
	   1 for 'abs(changePercent)>2.5&price>7.99'
	   2 for 'abs(changePercent)>5&price>7.99'
	   3 for 'abs(changePercent)>2.5&price>4.99'
	   OR a string such as 'abs(changePercent)>2.5&price>4.99'
	   default for None
	also see
	  from yh_chart import yh_quote_comparison; help(yh_quote_comparison)
	Note,
	  Info are pulled from DB 1st if last update was with in 'mmGap' minutes
	  If live data are not available, existed DB info will be used
	'''
	#from yh_predefined import yh_predefined_query
	tablename = "yh_{}".format(scrIds)
	cdt=s2dt()
	try:
		df,_,_ = find_mdb(dbname=dbname,tablename=tablename,dfTF=True)
		if len(df)>0 and 'pbdt' in df:
			t1=df['pbdt'].iloc[0]
			mmPassed=pd.Timedelta(cdt - t1).total_seconds() / 60.0
			if mmPassed>mmGap: # check if DB info is winthin last 'mmGap' (30-minute)
				df=[]
		else:
			df=[]
		if len(df)>0:
			if debugTF:
				sys.stderr.write("===Use {} from MDB:{}\n".format(scrIds,tablename))
			return df
		df=yh_predefined_query(scrIds,dfTF=True)
		if len(df)<1: # using whatever in the DB if live pulling failed
			df,_,_ = find_mdb(dbname=dbname,tablename=tablename,dfTF=True)
			return df
		df['pbdt'] = cdt
		if mappingTF:
			secInfo=sqlQuery('select * from mapping_ticker_cik where act_code=1')
			df = df.loc[df['ticker'].isin(secInfo['ticker'])]
		if addiFilter:
			if addiFilter==1:
				addiFilter='abs(changePercent)>0.5&price>7.99'
			elif addiFilter==2:
				addiFilter='abs(changePercent)>5&price>7.99'
			elif addiFilter==3:
				addiFilter='abs(changePercent)>2.5&price>4.99'
			if debugTF:
				sys.stderr.write("==addiFilter:{}\n".format(addiFilter))
			df = df.query(addiFilter)
		#df['pbdt'] = cdt
		df = df.merge(secInfo[['ticker','company_cn','sector','sector_cn']],on='ticker')
		df.reset_index(drop=True,inplace=True)
		if debugTF:
			sys.stderr.write("{}\n".format(df))
		if saveDB:
			clientM=None
			mobj, clientM, _ = write2mdb(df,clientM,dbname=dbname,tablename=tablename,zpk={'*'})
			sys.stderr.write("Data saved to {}::{}".format(clientM,tablename))
	except Exception as e:
		sys.stderr.write("**ERROR: bb_predefined(): {}\n".format(str(e)))
		df=[]
	return df.iloc[:6]

def screener_output_1(jobj):
	d = dict(ticker=jobj['symbol'])
	d.update(changePercent=jobj['regularMarketChangePercent']['raw'])
	if 'marketCap' in jobj:
		d.update(marketCap=jobj['marketCap']['raw'])
	else:
		d.update(marketCap=0)
	d.update(volume=jobj['regularMarketVolume']['raw'])
	d.update(price=jobj['regularMarketPrice']['raw'])
	d.update(change=jobj['regularMarketChange']['raw'])
	d.update(company=jobj['shortName'])
	return d

def yh_predefined(scrIds='most_actives',debugTF=False,**optx):
	''' 
	return yahoo screener info in JSON based on 'scrIds'
	'''
	urx = 'https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=true&lang=en-US&region=US&scrIds={scrIds}&start=0&count=25&co'
	url=urx.format(scrIds=scrIds)
	ret = requests.get(url,timeout=3)
	jdTmp = ret.json()
	return jdTmp


def yh_predefined_query(scrIds='most_actives',dfTF=False,saveDB=False,screenerTF=True,debugTF=False,**optx):
	''' 
	return yahoo screener DictList/DataFrame relavant info based on 'scrIds'
	'''
	try:
		jdTmp = yh_predefined(scrIds=scrIds)['finance']['result'][0]['quotes']
	except Exception as e:
			sys.stderr.write("**ERROR: {}: {}\n".format("yh_predefined_query", str(e)))
			return []
	dd=[]
	for jobj in jdTmp:
		try:
			if screenerTF:
				d = screener_output_1(jobj)
			else:
				d = jobj
			dd.append(d)
		except Exception as e:
			sys.stderr.write("**ERROR: {}: {}\n".format("yh_predefined_query", str(e)))
			continue
	if dfTF:
		dd = pd.DataFrame(dd)
	return dd

if __name__ == '__main__':
	description="Get predefined data via query2.finance.yahoo.com/v1/finance/screener"
	opts, args = parse_opt(sys.argv, description=description)
	if len(args)<1:
		args.append('most_actives')
	dd = bb_predefined(args[0],**opts)
	sys.stderr.write("\n{}\n".format(dd.to_string()))
