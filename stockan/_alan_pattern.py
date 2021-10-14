#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" OHLC Analysis
	Usage of:
	printf "select ticker from mapping_ticker_cik order by ticker" | /apps/fafa/bin/psql.sh -d eSTAR_2 -At | ohlc_analysis.py
	save winbgn as the 1st day of the pattern
	save signal as the last day of the pattern
	Created on Fri Dec 29 14:14:32 2017
	Last mod., Sun Jan  7 23:46:21 EST 2018

	uptrend defined as: close(t-2) >= 5-day-MA(t-2)
	downtrend defined as: close(t-2) < 5-day-MA(t-2)

	2-day patterns:
	bullish harami:
		1. clop 2-day sign changes: -+
		2. close > (close+open)/2 of t-1
		3. downtrend case of t-2

	bearish harami:
		1. clop 2-day sign changes: +-
		2. close < (close+open)/2 of t-1
		3. uptrend case of t-2

	3-day patterns:
	morning-star: clop 3-day sign changes:
		1. for (t-2,t-1,t) as -++
		2. close > (close+open)/2 of t-2
		3. downtrend case of t-2
	evening-star: clop 3-day:
		1. for (t-2,t-1,t) as +--
		2. close < (close+open)/2 of t-2
		3. uptrend case of t-2
	3-white-soldiers: clop 3-day:
		1. for (t-2,t-1,t) as +++
		2. clip( open(t-1), open(t), close(t-1) )
		3. downtrend case of t-2
	3-black-crows: clop 3-day:
		1. for (t-2,t-1,t) as ---
		2. clip( close(t-1), open(t), open(t-1) )
		3. uptrend case of t-2
	List of Functions:
	-def cmp_sign(a,b,tf):
	def clip_sign(amin,a,amax,tf):
	def get_ptn_index(vclsgn,ptn):
	def verify_conditions_msg(vma,vclose,vopen,vclsgn,ptn,isUpTick,isUpTrend,name,np,j):
	def verify_condition2_ptn(name,j,np,isUpTick,vclose,vopen,prnFlg=False):
	def calc_ptn_3days(vma,vclose,vopen,vclsgn,ptn,isUpTick,isUpTrend,name):
	def calc_ohlc_pattern(data,ptnLst=None):
	def add_MACD_pattern(data,ptnLst=[]):
"""
import sys
def cmp_sign(a,b,tf):
	xtf=True if a>=b else False
	return xtf if tf is True else not xtf

def clip_sign(amin,a,amax,tf):
	xtf=True if a>=amin and a<=amax else False
	return xtf if tf is True else not xtf

# DEPRECATED built-in calc_ptn_3days() condition 1
def get_ptn_index(vclsgn,ptn):
	np=len(ptn)
	# ptn forward counting
	winbgn = filter(lambda j: vclsgn[j:j+np]==ptn,  range(0,len(vclsgn)-np) )
	# ptn backward counting
	#winbgn = filter(lambda j: vclsgn[j-np+1:j+1]==ptn,  range(np-1,len(vclsgn)-1) )
	return winbgn

def verify_conditions_msg(vma,vclose,vopen,vclsgn,ptn,isUpTick,isUpTrend,name,np,j):
	print >>sys.stderr, "--Verifying {} index:\ncondition 1:\n  {}=={}".format(j,vclsgn[j:j+np],ptn)
	if name[-5:] == '_star' or name[-7:] == '_harami' :
		print >>sys.stderr, "condition 2:\n  close_{}[{}]>50%body[{}] should be {}".format(np-1,vclose[j+np-1], (vclose[j]+vopen[j])/2,isUpTick)
	elif name == '3_white_soldiers':
		print >>sys.stderr, "condition 2:\n  clip(open_0[{}],open_1[{}],close_0[{}]) should be {}".format(vopen[j],vopen[j+1],vclose[j],isUpTick)
		print >>sys.stderr, "\n  clip(open_1[{}],open_2[{}],close_1[{}]) should be {}".format(vopen[j+1],vopen[j+2],vclose[j+1],isUpTick)
	elif name == '3_black_crows':
		print >>sys.stderr, "condition 2:\n  clip(close_0[{}],open_1[{}],open_0[{}]) should be {}".format(vclose[j],vopen[j+1],vopen[j],not isUpTick)
		print >>sys.stderr, "\n  clip(close_1[{}],open_2[{}],open_1[{}]) should be {}".format(vclose[j+1],vopen[j+2],vopen[j+1],not isUpTick)
	print >>sys.stderr, "condition 3:\n  close[{}]>MA5[{}] should be {}".format(vclose[j],vma[j],isUpTrend)

def verify_condition2_ptn(name,j,np,isUpTick,vclose,vopen,prnFlg=False):
	tf=False
	if name[-5:] == '_star' or name[-7:] == '_harami' :
		tf=cmp_sign(vclose[j+np-1], (vclose[j]+vopen[j])/2,isUpTick)
	elif name == '3_white_soldiers':
		if clip_sign(vopen[j],vopen[j+1] ,vclose[j],isUpTick) and clip_sign(vopen[j+1],vopen[j+2] ,vclose[j+1],isUpTick):
			tf=True
	elif name == '3_black_crows':
		if clip_sign(vclose[j],vopen[j+1] ,vopen[j],not isUpTick) and clip_sign(vclose[j+1],vopen[j+2] ,vopen[j+1],not isUpTick):
			tf=True
	return tf

def calc_ptn_3days(vma,vclose,vopen,vclsgn,ptn,isUpTick,isUpTrend,name,debugTF=False):
	""" Calc forward counting position index for 3_white_soldiers & 3_black_crows
	    Inputs:
		vma: 5-day moving-average [close] price (default to 5-day M.A.)
		vclose: [close] price
		vopen: [open] price
		vclsgn: positive/negative sign of [close-open] price for [1 if x>0 else -1] condition
		ptn: sign pattern vector, e.g., [-1,1,1] as morning_star
		3-white-soldiers: clop 3-day: 
			1. for close-open:[clop] at [j,j+1,j+2] as +++
			2. clip( open[j], open[j+1], close[j] and  clip( open[j+1], open[j+2], close[j+1] )
			3. close[j] < ma[j] at downtrend case as [isUpTrend is False]
		3-black-crows: clop 3-day: 
			1. for close-open:[clop] at [j,j+1,j+2] as ---
			2. clip( close[j], open[j+1], open[j] and  clip( close[j+1], open[j+2], open[j+1] )
			3. close[j] < ma[j] at downtrend case as [isUpTrend is True]
		morning-star pattern conditions: 
			1. for close-open:[clop] 3-day sign pattern [ptn] at [j,j+1,j+2] as -++ 
			2. close[j+2] > (close[j]+open[j])/2  as [isUpTick is True]
			3. close[j] < ma[j] at downtrend case as [isUpTrend is False]
	    return a list of forward counting index 
	"""
	winbgn=[] # initialize winbgn
	winsig=[] # initialize winsig
	np=len(ptn) # length of pattern
	jx=0 # new winbgn tracker
	nc=len(vclsgn)
	for j in range(0,nc-np) :
		prnFlg=True if jx<1 else False
		if vclsgn[j:j+np]!=ptn: # check if condition 1 is not satisfied
			continue
		elif not cmp_sign(vclose[j],vma[j],isUpTrend) : # check if condition 3 is not satisfied
			continue
		elif verify_condition2_ptn(name,j,np,isUpTick,vclose,vopen,prnFlg) : # check condition 2
			winbgn.append(j) # save the 1st day of the pattern
			#winbgn.append(j+np-1) # save the last day of the pattern
			if(j+np<nc):
				winsig.append(j+np-1) # save signal as the last day of the pattern
			if jx<1 and debugTF:
				verify_conditions_msg(vma,vclose,vopen,vclsgn,ptn,isUpTick,isUpTrend,name,np,j)
			jx += 1
	if len(winbgn)>1 and debugTF:
		j=winbgn[jx-1]
		#j=winbgn[jx-1]-np+1
		#j=winbgn[jx-1]-np
		verify_conditions_msg(vma,vclose,vopen,vclsgn,ptn,isUpTick,isUpTrend,name,np,j)
	
	return (winbgn,winsig)

def calc_ohlc_pattern(data,ptnLst=None,debugTF=False):
	""" Calculate OHLC 3-day pattern
	"""
	# Rearrange input data for needed columns: vclop,vclose,vopen,vma
	(vclop,vclose,vopen,vma)= data[['clop','close','open','ma5']].values.transpose()
	vclsgn = map(lambda x:1 if x>0 else -1,vclop)
	if ptnLst is None:
		ptnLst=[
			{"name":"bullish_harami", "description":"Bullish Harami","pattern":[-1,1] ,"isUpTick":True,"isUpTrend":False},
			{"name":"bearish_harami", "description":"Bearish Harami","pattern":[1,-1],"isUpTick":False,"isUpTrend":True},
			{"name":"morning_star", "description":"Morning Star","pattern":[-1,1,1] ,"isUpTick":True,"isUpTrend":False},
			{"name":"evening_star", "description":"Evening Star","pattern":[1,-1,-1],"isUpTick":False,"isUpTrend":True},
			{"name":"3_white_soldiers", "description":"Three White Soldiers","pattern":[1,1,1],"isUpTick":True,"isUpTrend":False},
			{"name":"3_black_crows", "description":"Three Black Crows","pattern":[-1,-1,-1],"isUpTick":False,"isUpTrend":True}
		]
	for opx in ptnLst :
		if debugTF:
			print >>sys.stderr,"===RUNNING {} for pattern: {}".format(str(data['ticker'][0]),opx["name"])
		(opx["winbgn"],opx["winsig"])=calc_ptn_3days(vma,vclose,vopen,vclsgn,opx["pattern"],opx["isUpTick"],opx["isUpTrend"],opx["name"])
		opx["asof"]=int(data['pbdate'][-1:].values[0])
		opx["ticker"]=str(data['ticker'][0])
		ns=len(opx["winsig"])
		opx["nsig"]=ns
		np=len(opx["pattern"])
		opx["nwin"]=np
		# get correspondent yyyymmdd and ticker
		if ns>0 :
			opx["sigdate"]=list(data['pbdate'][opx["winsig"]].values)
			opx["windates"]=map(lambda x:list(data['pbdate'][range(x,x+np)].values),opx["winbgn"])
		else :
			opx["sigdate"]=[]
			opx["windates"]=[]
		if debugTF:
			print >>sys.stderr,"-- {} pattern:\n{}".format(opx["name"],opx)
			print >>sys.stderr,"-----------------------------------"
	return ptnLst

def add_MACD_pattern(data,ptnLst=[],debugTF=False):
	addLst=[ {"name":"MACD", "description":"Bullish MACD","pattern":[1]},
		{"name":"bearish_MACD", "description":"Bearish MACD","pattern":[-1]}]
	vs=data["signal_buysell_macd"]
	for opx in addLst :
		if debugTF:
			print >>sys.stderr,"===RUNNING {} for pattern: {}".format(str(data['ticker'][0]),opx["name"])
		opx["winsig"] = filter(lambda x: vs[x]/2==opx["pattern"][0],  range(len(vs)) )
		opx["winbgn"]=opx["winsig"]
		opx["asof"]=int(data['pbdate'][-1:].values[0])
		opx["ticker"]=str(data['ticker'][0])
		ns=len(opx["winsig"])
		opx["nsig"]=ns
		np=len(opx["pattern"])
		opx["nwin"]=np
		# get correspondent yyyymmdd and ticker
		if ns>0 :
			opx["sigdate"]=list(data['pbdate'][opx["winsig"]].values)
			opx["windates"]=map(lambda x:list(data['pbdate'][range(x,x+np)].values),opx["winbgn"])
		else :
			opx["sigdate"]=[]
			opx["windates"]=[]
		if debugTF:
			print >>sys.stderr,"-- {} pattern:\n{}".format(opx["name"],opx)
			print >>sys.stderr,"-----------------------------------"
		ptnLst.append(opx)
	return ptnLst
