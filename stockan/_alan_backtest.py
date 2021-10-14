#!/usr/bin/env python
""" PnL calculation for backtesting
    Program: "ara_pnl_backtest.py"
    sample data table: ara_outlook_factor_hist
    Usage of:
	printf "select distinct ticker from ara_outlook_factor_temp " | psql.sh -d ara -h vm2 -At |  ara_pnl_backtest.py --end=2017-09-01
    Last Mod., Thu Dec 14 11:28:50 EST 2017
"""
import sys
from optparse import OptionParser
import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def isNaN(num):
	return num != num

def pnl_calc(amount,price,xamt,pricm,xcost,expense=0) :
	""" PnL calc period
	    return (cost,proceeds,profit&loss)
	"""
	cost=amount*price+expense
	cum_cost=xcost+cost
	cum_amt=amount+xamt
	cum_prcds=cum_amt*pricm
	cum_pnl=cum_prcds-cum_cost
	return (cum_amt,cum_cost,cum_prcds,cum_pnl,cost)

def plus_only_position(amounts,xamt=10):
	""" default xamt to 10 units
	"""
	nc= len(amounts)
	tx=max(amounts[0],0)
	amounts[0]=tx
	for j in range(1,nc):
		ta=max(tx+amounts[j],0)
		ta=min(ta,xamt)
		amounts[j]=ta-tx
		tx=ta
	return amounts

def long_short_position(amounts,xfl=-1,xcap=1):
	""" default [-1,1]
	"""
	nc= len(amounts)
	tx=min(max(amounts[0],xfl),xcap)
	amounts[0]=tx
	for j in range(1,nc):
		ta=min(max(tx+amounts[j],xfl),xcap)
		amounts[j]=ta-tx
		tx=ta
	return amounts

def asg_strategy(confidence):
	nc= len(confidence)
	amts=np.arange(nc,dtype=float)
	for jx,x in enumerate(confidence):
		j=jx+1
		if j == len(confidence):
			continue
		if isNaN(x):
			amts[j]=0
		elif x <1: 
			amts[j]=-0.5
		elif x <1.5: 
			amts[j]=-0.25
		elif x <2.5: 
			amts[j]=-0.1
		elif x <3.5: 
			amts[j]=0.5
		elif x <4.5: 
			amts[j]=1
		elif x <6: 
			amts[j]=2
		else:
			amts[j]=0
	return amts

def cum_pnl_calc(amounts,prices,pricms) :
	""" PnL calculation for backtesting
	    INPUTS:
		prices: array of prices history during trade period
		amounts: array of traded amountss during trade period
		pricms: mark-to-market prices 
	    return pnl object
	"""
	nc=min(len(amounts),len(prices),len(pricms))
	(costs,prcds,pnls,cum_amounts,cum_costs,cum_prcds,cum_pnls)=np.array([np.arange(nc,dtype=float)]*7)
	for j in range(nc):
		if j>0 :
			xcost=cum_costs[j-1]
			xamt=cum_amounts[j-1]
		else:
			xcost=0
			xamt=0
		(cum_amounts[j],cum_costs[j],cum_prcds[j],cum_pnls[j],costs[j]) = pnl_calc(amounts[j],prices[j],xamt,pricms[j],xcost)

	prcds[1:]= np.diff(cum_prcds)
	pnls[1:]= np.diff(cum_pnls)
	df = pd.DataFrame(data=np.column_stack((pricms,amounts,costs,prcds,pnls,cum_amounts,cum_costs,cum_prcds,cum_pnls)),
		columns=["pricms","amounts","costs","prcds","pnls","cum_amounts","cum_cost","cum_prcds","cum_pnls"])
	
	return df


def tst_pnl_calc1():
	smpl=pd.read_csv("pnl_calc.02.tsv",sep="\t")
	smpl.fillna(0,inplace=True)
	smpl['amount']=smpl['Long']+smpl['modelShort']
	trade=smpl[['mtmPrice','amount','longPrice']]

	trade_price=trade['longPrice']
	trade_amount=trade['amount']
	mtm_price=trade['mtmPrice']
	trade_amount=plus_only_position(trade_amount.values)
	pnl=cum_pnl_calc(trade_amount,trade_price,mtm_price)
	print pnl
	#print >> sys.stderr, 'Total PnL {cum_pnls}'.format(**pnl)
	return pnl

def smpl_pnl_calc(smpl,xfl=-1,xcap=1):
	(trade_price,mtm_price,trade_amount)=smpl[['cost','price','amount']].values.transpose()
	trade_amount=long_short_position(trade_amount,xfl,xcap)
	pnl=cum_pnl_calc(trade_amount,trade_price,mtm_price)
	df = pd.concat([smpl,pnl],axis=1)
	#print >> sys.stderr, pd.concat([smpl,pnl],axis=1).to_csv(sep="\t")
	return df

def ara_pnl_calc(engine,ticker,pbstart,pbend) :
	sqTmp="""select confidence,s.* from (select ticker,confidence,asof from ara_outlook_factor_hist a where ticker='{0}' and factor='overall' and asof>={1} and asof<={2} order by asof) as a RIGHT JOIN (SELECT ticker, price, pbdate from spdr_price_hist  where pbdate>={1} and pbdate<={2} and ticker='{0}') as s on a.ticker=s.ticker and a.asof=s.pbdate ORDER BY s.pbdate
"""
	sqr=sqTmp.format(ticker,pbstart,pbend)
	#sqr=sqTmp.replace('{}',ticker)
	smpl = pd.read_sql(sqr,con=engine)
	trade_price=smpl['price']
	mtm_price=smpl['price']
	trade_amount=asg_strategy(smpl.confidence.values)
	trade_amount=plus_only_position(trade_amount)
	pnl=cum_pnl_calc(trade_amount,trade_price,mtm_price)
	df = pd.concat([smpl,pnl],axis=1)
	#print >> sys.stderr, pd.concat([smpl,pnl],axis=1).to_csv(sep="\t")
	return df

def ara_pnl_batch(tkLst,start,end,wmode):
	if end is None:
		end=datetime.datetime.now()
	else :
		end=datetime.datetime.strptime(end,'%Y-%m-%d')
	if start is None:
		start=end - datetime.timedelta(days=90) #- 90 days from end-date
	else :
		start=datetime.datetime.strptime(start,'%Y-%m-%d')
	pbstart=start.strftime('%Y%m%d')
	pbend=end.strftime('%Y%m%d')
	print  >> sys.stderr, 'Pulling data {0} from {1} to {2}'.format(tkLst,pbstart,pbend)

	engine = create_engine('postgresql://sfdbo@localhost:5432/ara')
	#wmode='replace'
	tableName='ara_backtest'
	for tkX in tkLst:
		try:
			df=ara_pnl_calc(engine,tkX,pbstart,pbend)
		except:
			print >> sys.stderr, "skip {0}".format(tkX)
			continue
		df['pbstart']=int(pbstart)
		df['pbend']=int(pbend)
		print >> sys.stderr, df.tail(1).to_csv(sep="\t")
		if wmode in ["replace","append"] :
			df.to_sql(tableName, engine, schema='public', index=False, if_exists=wmode)
			wmode='append'
		else:
			print >> sys.stderr, "No DB save, show on stdout"
			print >> sys.stdout, df.to_csv(sep="\t")

def calc_ohlc_pnl(data,ret,prd=125,xfl=-1,xcap=1):
	""" PnL calc period (days)
	"""
	smpl=data[['price','pbdate']]
	ns=len(smpl)
	smpl.index=range(ns)
	smpl.is_copy = False
	tablename="ohlc_pnl"
	dm=pd.DataFrame()
	lsc={'long':[0,xcap],'short':[xfl,0],'combo':[xfl,xcap]}
	for j in range(4) :
		print >> sys.stderr, j, ret
		smpl['amount']=0
		jx=j*2
		smpl['ticker']=ret[jx]['ticker']
		smpl['name']=ret[jx]['name']
		if len(ret[jx]['signal'])>0 :
			smpl.loc[ret[jx]['signal'],'amount']=2
		if len(ret[jx+1]['signal'])>0 :
			smpl.loc[ret[jx+1]['signal'],'amount']=-2
		smpl.loc[:,'cost']= smpl['price'].values
		mx=smpl.tail(prd).reset_index(drop=True)
		nm=len(mx)
		for (lx,ly) in lsc.iteritems() :
			mx['lsc']=[lx]*nm
			(xf,xc)=ly
			print >> sys.stderr, lx, xf,xc
			df=smpl_pnl_calc(mx,xf,xc)
			dm=pd.concat([dm,df])
	"""
	# DEPRECATED, use add_MACD_pattern() to run MACD
	smpl['name']="MACD"
	#smpl.rename(columns={"signal_buysell_macd":"amount"},inplace=True)
	smpl['amount']=data["signal_buysell_macd"].values
	mx=smpl.tail(prd).reset_index(drop=True)
	nm=len(mx)
	for (lx,ly) in lsc.iteritems() :
		mx['lsc']=[lx]*nm
		(xf,xc)=ly
		df=smpl_pnl_calc(mx,xf,xc)
		dm=pd.concat([dm,df])
	"""
	dm=dm.reset_index(drop=True)
	print >> sys.stderr, dm.head(2)
	print >> sys.stderr, dm.tail(2)
	return dm

def opt_ara_pnl():
	parser = OptionParser(version="%prog 0.5",
		description="ALAN PnL backtest calculation",
		usage="usage: %prog [option] SYMBOL1 ...  OR\n\tpipe SYMBOL input | %prog [option]")
	parser.add_option("-s","--start",action="store",dest="start",
		help="start from YYYY-MM-DD (default: 90-day from END)")
	parser.add_option("-e","--end",action="store",dest="end",
		help="end at YYYY-MM-DD (default: today)")
	parser.add_option("-w","--wmode",action="store",dest="wmode",default="replace",
		help="database table write-mode [replace|append|fail] (default: replace)")
	(options, args) = parser.parse_args()
	return (vars(options), args)

if __name__ == '__main__':
	(options, args)=opt_ara_pnl()
	if len(args) == 0 :
		print >> sys.stderr,"\nRead from pipe\n\n"
		tkLst = sys.stdin.read().strip().split("\n")
	else:
		tkLst = args[0:]
	ret = ara_pnl_batch(tkLst,options['start'],options['end'],options['wmode'])
