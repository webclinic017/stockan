#!/usr/bin/env python3
''' Calc intraday 3-major indices movement and most up/down tickers 
of either SP500 with marketcap>50B or DOW30 
and then create a market comment base on jinjia2: 'intraday_headline_cn.j2'
Usage of:
python3 -c "from headline_calc import headline_calc;dd=headline_calc(deltaTolerance=86400);print(dd)"
DB tables required:
PGDB:: sp500_component, dow_component, yh_quote_curr, 
MDG:: yh_quote_curr, market_indicator_quote
Last Mod.,
Fri Jul 30 21:05:01 EDT 2021
'''
import sys
import pandas as pd
from _alan_calc import sqlQuery,subDict

def  get_eqtLst(minMarketCap=50000000000):
	'''
	return equity list for import highlite based on dow_component,sp500_component, yh_quote_curr
	and minimum size (50B) of marketCalp
	'''
	xqTmp = '''SELECT a.ticker FROM sp500_component a, yh_quote_curr b 
		WHERE a.ticker=b.ticker AND 
		(b."marketCap">{} OR a.ticker in (SELECT ticker FROM dow_component))'''
	xqr = xqTmp.format(minMarketCap)
	try:
		eqtLst =  list(sqlQuery(xqr)['ticker'])
	except Exception as e:
		eqtLst=['AAPL','ABT','ACN','ADBE','AMGN','AMZN','AVGO','AXP','BA','BAC','BRK-B','C','CAT','CMCSA','COST','CRM','CSCO','CVX','DHR','DIS','DOW','FB','GOOG','GOOGL','GS','HD','HON','IBM','INTC','JNJ','JPM','KO','LIN','LLY','LMT','MA','MCD','MDT','MMM','MRK','MSFT','NEE','NFLX','NKE','NVDA','ORCL','PEP','PFE','PG','PM','PYPL','SBUX','T','TMO','TRV','TXN','UNH','UNP','UPS','UTX','V','VZ','WBA','WFC','WMT','XOM']
	return eqtLst

def headline_calc(tkLead='',t2Lead='^DJI',idxLst=None,eqtLst=None,np=3,xCol='changePercent',colLst=[],thd=0.05,deltaTolerance=300):
	'''
	return object of {'topLst', 'indexOrder', 'topIndex', 'indexLst', 'topUpDn'}
	Where
	  topIndex: ticker name of lead index defined as 'tkLead'='^GSPC'
	  topUpDn: sign in string UP/FLAT/DOWN within the range of 'thd'=[0.05,-0.05]
	  allUpDn: 1,0,-1 indecis all up/TBD/down
	  topLst: selected 'eqtLst' stock quote info ranked via 'changePercent'
	      w.r.t. the 'sign'/'topUpDn' of 'topIndex'
	  bttmLst: selected 'eqtLst' stock quote info oppsite to topLst
	  indexLst: 'idxLst' stock  quote info listed in the order of 'indexOrder'

	  Note aht topIndex quote info should be in the 'indexLst'
	'''
	from _alan_str import udfStr,find_mdb
	if eqtLst is None or len(eqtLst)<1:
		eqtLst = get_eqtLst()
	if idxLst is None or len(idxLst)<1:
		idxD = {"^GSPC":"標普500","^DJI":"道瓊","^IXIC":"納斯達克"}
		idxLst = list(idxD.keys())
	if colLst is None or len(colLst)<1:
		#colLst=['open','high','low','close','volume','ticker','change','changePercent','pbdate']
		colLst=['close','volume','ticker','change','changePercent','pbdate','pbdt']
	#xqTmp="SELECT * from yh_quote_curr WHERE ticker in ('{}')"

	# get indices quote performance
	#tkStr = "','".join(idxLst)
	#idxRtn = sqlQuery(xqTmp.format(tkStr))[colLst]

	# check if the quote is stale (longer than 300 seconds)
	jobj = dict(ticker={'$in':idxLst})
	idxRtn = find_mdb(jobj,dbname='ara',tablename='market_indicator_quote',dfTF=True)[0][colLst]
	for ky,val in idxD.items():
		try:
			ix = idxRtn.loc[idxRtn['ticker']==ky].index[0]
			idxRtn.loc[ix,'company_cn']=val
		except:
			continue
	if tkLead is None or len(tkLead)<1:
		tkIdx=(abs(idxRtn['changePercent'])).sort_values(ascending=False).index
		tkLead=idxRtn.loc[tkIdx[0],"ticker"]
		t2Lead=idxRtn.loc[tkIdx[1],"ticker"]
	else:
		tkLead='^GSPC'
	topDc=idxRtn.loc[idxRtn['ticker']==tkLead].iloc[0]
	pbdtMod=topDc['pbdt']
	deltaPassed=int(pd.Timedelta(pd.datetime.now()-pbdtMod).total_seconds())
	if deltaPassed>deltaTolerance:
		from yh_chart import yh_spark_hist as ysh;
		ysh(types='quote', debugTF=True)

	# get selected equities quote performance
	#tkStr = "','".join(eqtLst)
	#eqtRtn = sqlQuery(xqTmp.format(tkStr))[colLst]
	jobj = dict(ticker={'$in':eqtLst})
	eqtRtn = find_mdb(jobj,dbname='ara',tablename='yh_quote_curr',dfTF=True)[0][colLst]

	# calc 'topLst' w.r.t. the 'sign'/'topUpDn' of 'topIndex'
	pbdate=topDc['pbdate']
	chgPct=topDc[xCol]
	topUpDn = udfStr(chgPct,['UP','DOWN','FLAT'],thd)
	topSign = udfStr(chgPct,[1,0,-1],thd)
	sign = False if chgPct>=0 else True
	xd = eqtRtn.sort_values(by=[xCol],ascending=sign)
	leadLst = xd.iloc[:np]
	if(xd['changePercent'].iloc[0]*xd['changePercent'].iloc[-1])<0:
		bttmLst = xd.iloc[-1:]
	else:
		bttmLst = []

	# update my lead index in the top level
	dd = dict(topIndex=tkLead)
	dd.update(t2Index=t2Lead)
	dd.update(topUpDn=topUpDn)

	# add  all indices info to idxLst
	dd.update(indexLst=idxRtn.to_dict(orient='records'))
	indexOrder=[x['ticker'] for x in dd['indexLst']]
	dd.update(indexOrder=indexOrder)

	# determine if indices are all Up/Undetermined/Down
	if all([x['changePercent']<0 for x in dd['indexLst']]):
		allUpDn = -1
	elif all([x['changePercent']>0 for x in dd['indexLst']]):
		allUpDn = 1
	else:
		allUpDn = 0
	dd.update(allUpDn=allUpDn)

	# add  topLst 
	if len(leadLst)>0:
		dd.update(topLst=leadLst[colLst].to_dict(orient='records'))
	else:
		dd.update(topLst=[])
	if len(bttmLst)>0:
		dd.update(bttmLst=subDict(bttmLst,colLst).to_dict(orient='records'))
	else:
		dd.update(bttmLst=[])

	# get hiloRecord (based on past 1-year daily change since end date)
	hiloRecord=find_hiloRecord(ticker=tkLead,end=pbdate,days=366)
	dd.update(hiloRecord=hiloRecord)
	dd.update(start=pbdate)
	dd.update(mp3YN=False)

	return dd

def find_hiloRecord(ticker='^GSPC',end=None,days=366,debugTF=False):
	#from record_hilo import find_record_hilo as frh
	from record_hilo import recordHiLo as frh
	from _alan_calc import pull_stock_data as psd
	df=psd(ticker,end=end,days=days,pchgTF=True)
	endDT=df.index[-1]
	jobj=frh(df,endDT,ticker,debugTF=debugTF)
	hiloRecord = jobj['YTD'] if jobj['YTD'] else {}
	return(hiloRecord)

import numpy as np
import datetime
if __name__ == '__main__':
	from _alan_str import jj_fmt
	cdt=datetime.datetime.now()
	hm=int(cdt.strftime("%H00"))
	end_hm=np.clip(hm,900,1600)
	dtr= 86400 if (hm<900 or hm>1600) else 300
	dd= headline_calc(deltaTolerance=dtr)
	sys.stderr.write("{},{}\n".format(end_hm,dd))
	print(jj_fmt("{% include 'intraday_headline_cn.j2' %}",dd,dirname='templates/',outdir="US/mp3_hourly/",end_hm=end_hm ) )
	#print(jj_fmt("{% include 'intraday_briefing_cn.j2' %}",dirname='templates/',outdir="US/mp3_hourly/",end_hm=end_hm,**dd) )
