#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Description: plot time series from a csv file and summarise the graphics
    Usage of:
	csv2plotj2ts.py file --sep=DELIMITER
    Example:
	# minute data for 3 major market indices
	yh_hist_batch.py --range=1d --gap=1m   --no_database_save ^GSPC ^DJI ^IXIC --output=csv | python3 csv2plotj2ts.py  --columns=close,epochs,ticker  --xaxis=epochs --pivot_group=ticker --pivot_value=close  --title='美股大盤走勢2019-09-17' --return_since_inception --x_fmt='%H:%M' --extra_xs='pltStyle="classic";renColumns={"^GSPC":"標普500","^DJI":"道瓊指數","^IXIC":"納斯達克指數"}' -
	# OR (directly pulling data)
	python3 csv2plotj2ts.py TCEHY ^HSI ^GSPC --columns=close,pbdate,ticker  --xaxis=pbdate --pivot_group=ticker --pivot_value=close  --title='騰訊/恆生/標普-2021年初至今報酬率' --interpolate --return_since_inception --x_fmt='%m/%d/%y' --extra_xs='ranged="20210101,20210804";gap="1d";pltStyle="classic";renColumns={"TCEHY":"騰訊","^HSI":"恆生指数","^GSPC":"標普500"}'
	python3 csv2plotj2ts.py ^GSPC ^DJI ^IXIC --columns=close,epochs,ticker  --xaxis=epochs --pivot_group=ticker --pivot_value=close  --title='美股大盤走勢' --return_since_inception --x_fmt='%H:%M' --extra_xs='ranged="1d";gap="1m";pltStyle="classic";renColumns={"^GSPC":"標普500","^DJI":"道瓊指數","^IXIC":"納斯達克指數"}' 
	# FROM data file
	# OR (near realtime data)
	yh_hist_batch.py --range=1d  --no_database_save AAPL --output=csv| csv2plotj2ts.py - --columns=close,epochs  --xaxis=epochs --title=Apple
	#== FOR DAILY DAT
	# directly pull data from web and plot 2y-axis via sharex=True
	python3 csv2plotj2ts.py ^GSPC ^VIX --columns=ticker,close,epochs  --xaxis=epochs --pivot_group=ticker --pivot_value=close  --title='Market vs. Volatility' --trendline --extra_xs='sharexTF=True;ranged="3mo";gap="1d"'
	# OR directly pull data from web and calc cumulated-returns of non 'columns2' list then plot 2y-axis via twinx()
	python3 csv2plotj2ts.py ^GSPC ^VIX --columns=ticker,close,epochs  --xaxis=epochs --pivot_group=ticker --pivot_value=close  --title='標普500累積獲利相對恐慌指數圖' --extra_xs='kind2="line";columns2="^VIX";ranged="3mo";gap="1d";renColumns={"^GSPC":"標普500","^VIX":"恐慌指數"};' --return_since_inception
	# OR (stock data with --src and intraday minute data with 2-axis)
	yh_hist_batch.py --range=1d --gap=1m   --no_database_save ^GSPC ^VIX  --output=csv | python3 csv2plotj2ts.py - --columns=ticker,close,epochs  --xaxis=epochs --pivot_group=ticker --pivot_value=close  --title='Market vs. Volatility' --extra_xs='sharexTF=True;'
	# OR (stock data with --src and intraday minute data)
	yh_hist_batch.py --gap=1d --range=3mo  --no_database_save AAPL --output=csv| csv2plotj2ts.py - --columns=open,close,pbdate  --xaxis=pbdate --title=Apple
	# OR (daily return since inception )
	yh_hist_batch.py --gap=1d --range=3mo  --no_database_save AAPL --output=csv| csv2plotj2ts.py - --columns=open,close,pbdate  --xaxis=pbdate --title=Apple --return_since_inception
	# OR (pivot data)
	printf "select m.label as ticker,p.close as price,p.pbdate from prc_hist p,mapping_series_label m where p.name in ('^GSPC','^TWII','000001.SS','^SOX','^DJI') and p.pbdate>20170101 and p.name=m.series order by m.label,p.pbdate" | psql.sh -d ara | grep -v rows  | python3 csv2plotj2ts.py --pivot_group=ticker --pivot_value=price  --title='Market Overview 2018-05-25' --interpolate --return_since_inception -
	# OR (pivot data and near realtime per minute)
	yh_hist_batch.py --range=1d  --no_database_save AAPL XLK SPY --output=csv| csv2plotj2ts.py - --columns=ticker,close,epochs  --xaxis=epochs --pivot_group=ticker --pivot_value=close  --title='Market Closing Overview' --interpolate --return_since_inception --trendline
	# OR (pivot data with minute data)
	csv2plotj2ts.py CRWD --extra_xs='columns2="volume";gap="1m";renColumns={"close":"CRWD"}' --columns=close,volume --xaxis=epochs --x_fmt='%H:%M' --title="CrowdStrike Intraday"
	# OR (fred data with --src)
	csv2plotj2ts.py DGS2 --src=fred --columns=close,pbdate
	# OR (stock data with --src and candlestick graph)
	csv2plotj2ts.py IBM --src=yh --columns=close,open,high,low,volume,pbdate --title="IBM OHLC" --days=90 --ohlc
	# OR get data from an CSV-formatted text-file
	python3 csv2plotj2ts.py AAPL_XLK_SPY.dat --src=filename --columns=ticker,close,epochs  --xaxis=epochs --pivot_group=ticker --pivot_value=close  --title='Market Closing Overview' --interpolate --return_since_inception --trendline
	# OR (minute data and candlestick graph)
	yh_hist_batch.py --gap=1m --range=1d --no_database_save --output=csv AAPL| csv2plotj2ts.py - --columns=close,open,high,low,volume,epochs,ticker --ohlc --title="Intraday AAPL OHLC"  --xaxis=epochs --trendline
	# OR (input minute data and candlestick Combo graph)
	yh_hist_batch.py --gap=1m --range=1d --no_database_save --output=csv AAPL| csv2plotj2ts.py - --columns=ticker,close,open,high,low,volume,epochs --ohlc_combo --title="Intraday AAPL"   --xaxis=epochs --trendline 
	# OR (src minute data and candlestick Combo graph)
	csv2plotj2ts.py AAPL --src=yh --extra_xs='ranged="1d";gap="1m"' --columns=close,open,high,low,volume,epochs --ohlc_combo --title="Intraday AAPL OHLC"  --xaxis=epochs --trendline
	# OR run specific function like "get_csv2plot"
	yh_hist_batch.py --gap=1m --range=1d --no_database_save --output=csv AAPL| csv2plotj2ts.py - --columns=close,open,high,low,volume,epochs,ticker --ohlc --title="Intraday AAPL OHLC"  --xaxis=epochs --trendline --extra_xs='run="get_csv2plot"' 

    Note: return_since_inception will use $1 as the initial investment if the initial is less than $1
    Last mod., Tue Nov  3 13:40:38 EST 2020
"""

import sys
from optparse import OptionParser
from scipy.interpolate import interp1d
import pandas as pd
import json
from datetime import datetime
from pandas.plotting import register_matplotlib_converters
from _alan_calc import pqint,getKeyVal,renameDict,subDF
register_matplotlib_converters()
if sys.version_info.major == 2:
	reload(sys)
	sys.setdefaultencoding('utf8')

def subDict(myDict,kyLst,reverseTF=False):
	if reverseTF is True: # invert-match, select non-matching [kyLst] keys 
		return { ky:myDict[ky] for ky in myDict.keys() if ky not in kyLst }
	else:
		return { ky:myDict[ky] for ky in myDict.keys() if ky in kyLst }

def ymd_parser(x,fmt='%Y%m%d'): return datetime.strptime(str(x),fmt)

def epoch_parser(x,s=1000): return datetime.fromtimestamp(int(x/s))

def extrapolate_series(yo):
	yg=yo.dropna()
	#ygi = [int(x) for x in yg.index.values]
	#fn = interp1d(ygi, yg.values, fill_value='extrapolate')
	#yoi = [int(x) for x in yo.index.values]
	#return fn((yoi)
	fn = interp1d(list(map(int,yg.index.values)), yg.values, fill_value='extrapolate')
	return fn(list(map(int,yo.index.values)))


def get_csvdata(args,sep='|',src=None,days=730,start=None,end=None,colLst=None,hdrLst=None,**optx):
	"""
	Get data in datafram with selected [colLst]
	"""
	if isinstance(args,pd.DataFrame):
		df = args
		if colLst is not None and df.size > 0:
			df =  df[ list(set(df.columns) & set(colLst.split(','))) ]
		if hdrLst is not None:
			xLst,yLst = hdrLst.split('=')
			xyD = dict(zip(xLst.split(','),yLst.split(',')))
			df.rename(columns=xyD,inplace=True)
		return df
	if len(args)<1:
		return None
	filename=args[0]
	if filename=='-':
		df=pd.read_csv(sys.stdin,sep=sep)
	elif src.upper()=='FILENAME':
		df=pd.read_csv(filename,sep=sep)
	elif len(args)>1 and src=='yh':
		# TBD, only pull one ticker at a time for now
		#from yh_chart import yh_spark_hist as ysh;
		#ranged=optx.pop('ranged','1d')
		#gap=optx.pop('gap','1m')
		#optx.update(range=ranged,interval=gap,dfTF=True,tsTF=True)
		#df=ysh(tkLst=args,types='chart',**optx)
		from yh_hist_batch import batch_yh_hist
		df = batch_yh_hist(args,days=days,src=src,start=start,end=end,**optx)
		if 'ticker' not in df and 'name' in df:
			df=renameDict(df,{"name":"ticker"})
	elif src is not None:
		from _alan_calc import pull_stock_data
		df = pull_stock_data(filename,days=days,src=src,start=start,end=end,**optx)
	else:
		df = pd.read_csv(filename,sep=sep)
	if df.size < 1:
		sys.stderr.write("**ERROR:{} @{}()".format('Data Not Found!','get_csvdata'))
		return {}
	if colLst is not None:
		df =  df[ list(set(df.columns) & set(colLst.split(','))) ]
	df.dropna(inplace=True)
	if hdrLst is not None:
		xLst,yLst = hdrLst.split('=')
		xyD = dict(zip(xLst.split(','),yLst.split(',')))
		df.rename(columns=xyD,inplace=True)
	return df

def vertex_mnmx(dh,prc_open):
	dmx = dh.loc[dh['actual'] == dh['actual'].max()].iloc[0].to_dict()
	dmn = dh.loc[dh['actual'] == dh['actual'].min()].iloc[0].to_dict()
	dk = []
	if dmx['iptday']>dmn['iptday']:
		dk = dk + [dmn, dmx]
	else:
		dk = dk + [dmx, dmn]
	if dk[0]['iptday'] > dh.loc[0,'iptday']: 
		dk = [dh.iloc[0].to_dict()] + dk
	if dk[-1]['iptday'] < dh['iptday'].iloc[-1]: 
		dk = dk + [dh.iloc[-1].to_dict()]	
	dk[0]['actual'] = prc_open
	dk = pd.DataFrame(dk)
	dk['pchg_open'] = [ x/prc_open-1 for x in dk['actual']]
	dk['sign'] = [ 1 if x>0 else -1  for x in dk['pchg_open']]
	dk['difday'] = dk['iptday'].diff()
	return dk

import matplotlib
if matplotlib.get_backend() != 'TkAgg':
	matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
from matplotlib.ticker import FuncFormatter
font_path = "/usr/share/fonts/truetype/arphic/uming.ttc"
prop = mfm.FontProperties(fname=font_path)

def fontProp(fname=font_path,**optx):
	return mfm.FontProperties(fname=fname,**optx)

def plot_csvdata(df, nbins=6,rsiYN=False,title=None,pivot_value=None,pivot_group=None,pngname=None,x_fmt="%b-%d-%y",interpolateYN=True,backend="tkAgg",npar=15,tsTF=True,xaxis=None,trendTF=False,hiLoTF=False,debugTF=False,ohlcTF=False,ohlcComboTF=False,lang='en',**kwargs):
	if debugTF:
		sys.stderr.write("===plot_csvdata VARS:\t{}\n".format(locals()))
	import matplotlib.dates as mdates
	import matplotlib.image as mimage
	import matplotlib.ticker as mticker
	#pltStyle=getKeyVal(kwargs,'pltStyle','dark_background')
	pltStyle=getKeyVal(kwargs,'pltStyle','classic')
	figsize=getKeyVal(kwargs,'figsize',(11,6))
	ylabel=getKeyVal(kwargs,'ylabel','')
	fontpath=getKeyVal(kwargs,'fontpath',None)
	trendName=getKeyVal(kwargs,'trendName',None)
	sharexTF=getKeyVal(kwargs,'sharexTF',False)
	if fontpath is not None:
		prop.set_file(fontpath)
	if pltStyle in plt.style.available:
		plt.style.use(pltStyle)
	#- Use backend to 'tkAgg' for cronjob
	if pngname is None or len(pngname)<=4:
		plt.switch_backend(backend)

	#- Create datetime index
	idxname='date'
	pbname=xaxis
	if isinstance(df.index,pd.DatetimeIndex):
		pass
	elif pbname in df.columns:
		sdate = str(df[pbname].iloc[0])
		if sdate.isdigit() == True:
			if int(sdate)>123456789:
				idxpt=[epoch_parser(x) for x in df[pbname]]
			else:
				idxpt=[ymd_parser(x,fmt="%Y%m%d") for x in df[pbname]]
		else:
			idxpt=[ymd_parser(x,fmt=x_fmt) for x in df[pbname]]
		df.set_index(pd.DatetimeIndex(idxpt),inplace=True)
		df.index.rename(idxname,inplace=True)
		df = df.drop(pbname,1)
	elif idxname in df.columns:
		df[idxname] = pd.to_datetime(df[idxname])
		df.set_index(idxname,inplace=True)
	else:
		df = df.reset_index(drop=True)


	#- Create a pivot table
	if pivot_group in df.columns and pivot_value in df.columns:
		df=df.pivot_table(index='date',columns=pivot_group,values=pivot_value)

	#- Create linear-interpolation for missing data 
	if interpolateYN is True:
		df=df.apply(extrapolate_series,axis=0)


	#- Create 2nd dataframe 'df2' for 2-Yaxis plot
	dux=getKeyVal(kwargs,'columns2',None)
	if debugTF:
		sys.stderr.write("DF:\n{}\n{}\n".format(df.tail(),df.shape))
	if dux is not None:
		colLst2=dux.split(',')
		df2=subDF(df,colLst2)
		df=subDF(df,colLst2,reverseTF=True)
	else:
		df2={}

	#- Create return since inception
	if rsiYN is True:
		de=[] 
		for j in range(df.shape[1]): 
			inix = df.iloc[0,j] if df.iloc[0,j]>1 else 1
			de.append(df.iloc[:,j]/inix*100.-100)
		#de = [df.iloc[:,j]/df.iloc[0,j]*100.-100 for j in range(df.shape[1])] 
		df = pd.concat(de,axis=1)

	#- Rename columns
	renColumns=getKeyVal(kwargs,'renColumns',{})
	if len(renColumns)>0:
		df = renameDict(df,mapper=renColumns)
		if len(df2)>0:
			df2 = renameDict(df2,mapper=renColumns)
		if debugTF:
			sys.stderr.write("After rename DF:\n{}\n{}:{}\n".format(df.tail(),df.columns,df.shape))

	#- NO PLOTTING, just return data
	if 'plotTF' in kwargs and kwargs['plotTF'] is False:
		return df,{},{}

	#- Create trend curve
	if trendTF:
		try:
			from _alan_pppscf import vertex_locator
			if trendName is None:
				trendName = df.columns[0]
			if debugTF is True:
				sys.stderr.write("{}\n{}\n".format("Trend data:",df[trendName]))
			dg, dh = vertex_locator(df[trendName],npar=npar,debugTF=debugTF)
			if debugTF is True:
				sys.stderr.write("{}\n{}\n".format("Trendline dg:",dg))
		except Exception as e:
			sys.stderr.write("**ERROR: {} @ {}\n".format(str(e),'trendline'))
			dg, dh = {},{}

	if hiLoTF:
		try:
			from _alan_calc import find_mnmx
			if trendName is None:
				trendName = df.columns[0]
			dg = pd.Series(df[trendName].iloc[sorted(set(find_mnmx(df[trendName])))],name='hiLo')
			if debugTF is True:
				sys.stderr.write("{}\n{}\n".format("hiLoline dg:",dg))
		except Exception as e:
			sys.stderr.write("**ERROR: {} @ {}\n".format(str(e),'hiLoline'))
			dg = {}

	if title is None: 
		title="/".join(df.columns).upper()
		if rsiYN is True:
			title += " Return Since Inception"

	#- plot simple line plot
	if tsTF is False:
		df = df.reset_index(drop=True)

	if debugTF is True:
		sys.stderr.write("{}\n".format(df.head()))
		sys.stderr.write("{}\n".format(df.tail()))
	nobs=len(df.index)
	nsp = int(nobs/nbins) if nobs>nbins*2 else nobs
	#ds=[y for j,y in enumerate(df.index) if j%nsp==0]
	#ax=df.plot(xticks=ds,title=title)
	colorUD = ['red','green'] if lang=='cn' else ['green','red']
	if ohlcComboTF is True:
		from alan_plot import plot_candlestickCombo
		from _alan_calc import run_tech
		chartType = 'minute' if pbname == 'epochs' else 'chart'
		#ma1=5;ma2=30
		ma1,ma2=sma=getKeyVal(kwargs,'sma',[5,30])
		datax = run_tech(df, pcol='close',winLst=sma,debugTF=debugTF,nanTF=True)
		if 'open' not in datax:
			return datax, None, None
		fig, axes = plot_candlestickCombo(datax,title,ma1,ma2,block=False,chartType=chartType,trendTF=trendTF,npar=npar,debugTF=debugTF,colorUD=colorUD,title=title)
		#plt.suptitle(title,fontsize=18,fontproperties=prop)
		if pngname is not None and len(pngname)>4 and '.' in pngname:
			ghLst = plt.gcf().canvas.get_supported_filetypes().keys()
			ghx = pngname.split('.')[-1]
			format = ghx.lower() 
			if ghx.lower() in ghLst:
				format = ghx.lower() 
			else:
				format = 'svg'
			pngname = pngname.replace(ghx,'svg')
			plt.savefig(pngname, format=format) #, bbox_inches='tight',dpi=1000)
		# skip the plot if pngname='noshow'
		elif pngname is None:
			plt.show(axes)
		return datax, fig, axes
	fig, ax=plt.subplots(figsize=figsize,sharex=sharexTF)
	if ohlcTF is True:
		if 'marketVolume' in df:
			df.rename(columns={'marketVolume': 'volume'},inplace=True)
		if 'open' not in df and 'close' in df:
			df['open']=df['high']=df['low']=df['close']
		elif 'open' not in df:
			return df, None, None
		from alan_plot import plot_candlestick
		chartType = 'minute' if pbname == 'epochs' else 'chart'
		ax = plot_candlestick(df,tsidx=df.index,chartType=chartType,title=title,block=False,debugTF=debugTF,ax=ax,trendTF=trendTF,npar=npar,colorUD=colorUD)
		x_fmt = "%H:%M" if chartType == 'minute' else x_fmt
	else:
		colorLst=['blue','red','green','salmon','lightgray','cyan']
		if len(df.columns)>1 and sharexTF:
			col2=df.columns[1]
			df.plot(ax=ax,grid=True,color=colorLst,secondary_y=col2)
			ylabel2=getKeyVal(kwargs,'ylabel2',col2)
			ax.right_ax.set_ylabel(ylabel2,fontproperties=fontProp(size=12))
		else:
			df.plot(ax=ax,grid=True,color=colorLst)
		#ax=df.plot(figsize=(11,6))
		ax.set_ylabel(df.columns[0])
		yfmt=getKeyVal(kwargs,'yfmt',',.0f')
		ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: '{:{}}'.format(x,yfmt)))
		if any([trendTF,hiLoTF]) and len(dg)>0:
                        dg.plot(ax=ax)
		if len(df2)>0:
			kind2=getKeyVal(kwargs,'kind2','area')
			color2=getKeyVal(kwargs,'color2','green')
			alpha2=getKeyVal(kwargs,'alpha2',0.4)
			yfmt2=getKeyVal(kwargs,'yfmt2',',.2f')
			ylabel2=getKeyVal(kwargs,'ylabel2','%')
			if debugTF:
				sys.stderr.write("DF2:\n{}\n{}\n".format(df2.tail(),df2.shape))
			axv = ax.twinx()
			df2.plot(ax=axv,kind=kind2,color=color2,alpha=alpha2,legend=False)
			axv.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: '{:{}}'.format(x,yfmt2)))
			if len(ylabel2)>0:
				axv.set_ylabel(ylabel2)
	if len(ylabel)>0:
		ax.set_ylabel(ylabel,fontproperties=fontProp(size=12))
	elif rsiYN is True: # calc Returns Since Incept
		ax.set_ylabel("Returns Since Inception (%)")
	ax.grid(linestyle='dotted',linewidth=0.5)
	if df.index._typ == "datetimeindex":
		mddfmt=mdates.DateFormatter(x_fmt)
		ax.xaxis.set_major_formatter(mddfmt)
		xtinterval=(df.index[1]-df.index[0])
		if xtinterval.days < 7 and  xtinterval.days>=1 : # daily data
			ax.set_xlim(df.index[0], df.index[-1])
			#ax.xaxis.set_major_locator(mdates.MonthLocator(interval=int(nsp/30.+0.97)))
			bymd = [1,5,10,15,20,25] if nobs<50 else [1,15] if nobs<120 else [1]
			itv = 1 if nobs<160 else int(nsp/30.+0.97)
			xlocator = mdates.MonthLocator(bymonthday=bymd,interval=itv)
			ax.xaxis.set_major_locator(xlocator)
			# check if min/max of xaxis should be included major ticks
			if debugTF is True:
				sys.stderr.write("{} {}\n".format( ax.get_xticks(),ax.get_xlim()))
			xtcks = list(ax.get_xticks())
			x1,x2 = xtcks[:2]
			xmin,xmax = ax.get_xlim()
			if (x1-xmin)>(x2-x1)*0.6:
				xtcks = [xmin] + xtcks
			if (xmax-xtcks[-1])>(x2-x1)*0.6:
				xtcks = xtcks + [xmax]
			ax.set_xticks(xtcks)
			ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
			if debugTF is True:
				sys.stderr.write("{}\n".format(ax.get_xticks()))
			sys.stderr.write("{}\n".format( "Daily data use MonthLocator"))
		elif xtinterval.seconds < 30: # second data
			locator = mdates.AutoDateLocator()
			locator.intervald[5] = [0,5,10,15,20,25,30,35,40,45,55]
			mddfmt = mdates.AutoDateFormatter(locator)
			mddfmt.scaled[1/(24.*60.)] = '%M:%S' 
			ax.xaxis.set_major_locator(locator)
			ax.xaxis.set_major_formatter(mddfmt)
			sys.stderr.write("{} {}\n".format( "Second data use AutoDateLocator",xtinterval.seconds))
		elif xtinterval.seconds < 100 : # minute data
			bym = [0,15,30,45] if nobs<=120 else [0,30] if nobs<=360 else [0]
			xlocator = mdates.MinuteLocator(byminute=bym, interval = 1)
			ax.xaxis.set_major_locator(xlocator)
			sys.stderr.write("{} {}\n".format( "Minute data use MinuteLocator",xtinterval.days))
		else: # periodic data
			sys.stderr.write("{} {}\n".format( "Periodic data use DayLocator",nsp))
			ax.xaxis.set_major_locator(mdates.DayLocator(interval=nsp))
	ax.xaxis.label.set_visible(False)
	plt.title(title,fontproperties=fontProp(size=30))
	plt.xticks(rotation='20',fontsize=12)
	if len(df.columns)>0 and sharexTF is True:
		h1, l1 = ax.get_legend_handles_labels()
		h2, l2 = ax.right_ax.get_legend_handles_labels()
		if len(ylabel)<1 and len(l1)>0:
			ylabel=l1[0]
		ax.legend(h1+h2, [ylabel]+[ylabel2], loc="upper left",prop=prop)
	elif len(df.columns)>0 and sharexTF is False and ohlcTF is False:
		ax.legend(loc="upper left",prop=prop)
	#ax.legend().set_visible(False)
	#logo = mimage.imread("aicaas_icon.png")
	#plt.figimage(logo, xo=20,yo=420)
	plt.subplots_adjust(left=0.1,bottom=0.30)
	if pngname is not None and len(pngname)>4:
		plt.savefig(pngname)#, bbox_inches='tight',dpi=1000)
		sys.stderr.write("Save chart {} to {}\n".format(title,pngname))
	else:
		print("===>ax:",ax)
		plt.show()
		#plt.show(ax)
	return df, fig, [ax]

def run_comment_hourly(df, opts=None):
	from _alan_str import generate_comment_hourly
	if 'ticker' not in opts:
		if 'ticker' in df.columns:
			opts['ticker'] = df['ticker'].iloc[0]
		else:
			return {}
	ohlcComboTF = opts['ohlcComboTF'] if 'ohlcComboTF' in opts else False
	jobj = {}
	if opts['j2ts'] is not None and len(opts['j2ts'])>0:
		ts=opts['j2ts']
	elif 'j2name' in opts and len(opts['j2name'])>0:
		fj2name = "{}_{}.j2".format(opts['j2name'],opts['lang'])
		ts='{} include "{}" {}'.format('{%',fj2name,'%}')
	else:
		ts=''
	if len(ts)<1:
		return {}
	try:
		jobj = generate_comment_hourly(ts,df,opts)
		ret = jobj['comment'] if 'comment' in jobj else ''
		if 'plotTF' in opts and opts['plotTF'] is False:
			return jobj
		if len(ret)>0:
			pqint( ret, file=sys.stdout)
			if 'plotTextTF' in opts and opts['plotTextTF']:
				bottom = 0.35 if ohlcComboTF is True else 0.25
				plt.subplots_adjust(left=0.1,bottom=bottom)
				txbtm = 0.05 if ohlcComboTF is True else 0.1
				plt.text(0.05,txbtm,ret,fontsize=11,color='yellow',fontproperties=prop, transform=plt.gcf().transFigure)
			pngname = opts['pngname']
			if pngname is not None and len(pngname)>4:
				#pngx = pngname.split('.')[0].split('_')[0] + opts['rpt_time'].strftime("_%Y%m%d_%H%M") +'.svg'
				#if opts['pngname'] == pngx:
				#	pqint( "svg filename {} is duplicated!".format(pngx), file=sys.stderr)
				#	return jobj
				#opts['pngname'] = pngx
				outdir = opts['outdir'] if 'outdir' in opts else '.'
				pngpath = "{}/{}".format(outdir,pngname)
				sys.stderr.write("rpt_time:{},svg_path:{}\n".format(opts['rpt_time'],pngpath))
				plt.savefig(pngpath, format='svg')
			else:
				plt.show()
	except Exception as e:
		sys.stderr.write("**ERROR: @ generate_comment_hourly\n{}\n".format(str(e)))
	return jobj

def get_csv2plot(args,opts=None,**optx):
	"""
	plot time series data from csv file 
	"""
	#- Set input parameters
	if opts is None or len(opts)<1:
		opts, _ = opt_csv2plot([])
	if optx is not None:
		opts.update(optx)
	debugTF=opts['debugTF']
	if debugTF:
		sys.stderr.write("===get_csv2plot VARS:\t{}\n".format(locals()))
	#for ky,va in opts.items():
	#	exec("{}=va".format(ky))

	#- Get data in datafram with selected [colLst]
	if isinstance(args,pd.DataFrame) is True: 
		df = args
	else:
		gkys=['sep','src','days','start','end','colLst','hdrLst','debugTF','ranged','gap','tsTF','pchgTF','searchDB','range','interval']
		df = get_csvdata(args, **subDict(opts,gkys))

	if df is None or len(df)<1 or df.size<1:
		return {},{},{}
	if debugTF is True:
		sys.stderr.write("OPTS: {}\n".format(opts))
		sys.stderr.write("DATA:\n{}\n".format(df.head()))

	pkys=['interpolateYN','title','debugTF','tsTF','rsiYN','pivot_value','pivot_group','ohlcTF','ohlcComboTF','xaxis','trendTF','pngname','x_fmt','nbins','npar','backend','lang']
	#df,fig,axes = plot_csvdata(df,**subDict(opts,pkys))
	df,fig,axes = plot_csvdata(df,**opts)
	return df,fig,axes

def run_csv2plot(args,opts=None,**optx):
	pngname=opts['pngname']
	opts['pngname']='noshow'
	df,fig,axes = get_csv2plot(args,opts=opts,**optx)
	opts['pngname']=pngname
	opts['intradayTF'] = True if (df.index[1]-df.index[0]).days<1 else False
	if 'rpt_time' not in opts:
		rpt_time = df.index[-1]
		opts.update(rpt_time=rpt_time)
	try:
		jobj = run_comment_hourly(df, opts)
	except Exception as e:
		sys.stderr.write("**ERROR:{} @{}\n".format(str(e),'run_csv2plot'))
		jobj = {}
	return df, opts, jobj

def opt_csv2plot(argv=[],retParser=False):
	""" command-line options initial setup
	    Arguments:
		argv:	list arguments, usually passed from sys.argv
		retParser:	OptionParser class return flag, default to False
	    Return: (options, args) tuple if retParser is False else OptionParser class 
	"""
	parser = OptionParser(usage="usage: %prog [option] FILENAME", version="%prog 1.0",
		description="Time-series Plotting Utility via matplotlib")
	parser.add_option("-s","--sep",action="store",dest="sep",default="|",
		help="field separator (default: |)")
	parser.add_option("","--xaxis",action="store",dest="xaxis",default="pbdate",
		help="x-axis [pbdate|epochs] column name (default: pbdate in yyyymmdd)")
	parser.add_option("","--columns",action="store",dest="colLst",
		help="selected columns (default: ALL)")
	parser.add_option("","--ren_header",action="store",dest="hdrLst",
		help="rename header columns")
	parser.add_option("-t","--title",action="store",dest="title",
		help="title (default: combo-colunms)")
	parser.add_option("-n","--nbins",action="store",dest="nbins",default="6",type=int,
		help="number of bins in x-axis (default: 6)")
	parser.add_option("","--return_since_inception",action="store_true",dest="rsiYN",default=False,
		help="use Return since Inception plot. Note: $1 will be used as the initial investment if the initial is less than $1")
	parser.add_option("","--interpolate",action="store_true",dest="interpolateYN",default=False,
		help="use linear-interplation for missing data")
	parser.add_option("","--pivot_group",action="store",dest="pivot_group",
		help="pivot table group by column, must pair with PIVOT_VALUE")
	parser.add_option("","--pivot_value",action="store",dest="pivot_value",
		help="pivot table display value column, must pair with PIVOT_GROUP")
	parser.add_option("","--x_fmt",action="store",dest="x_fmt",default='%m-%d-%y',
		help="graph x-axis format (default: %m-%d-%y)")
	parser.add_option("","--png",action="store",dest="pngname",
		help="graph name (default: None)")
	parser.add_option("","--backend",action="store",dest="backend",default='tkAgg',
		help="matplotlib new backend(default: tkAgg)")
	parser.add_option("","--no_time_series",action="store_false",dest="tsTF",default=True,
		help="Simple line plot no time-series")
	parser.add_option("-l","--lang",action="store",dest="lang",default="cn",
		help="language mode [cn|en] (default: cn), ohlc/ohlc_combo ONLY")
	parser.add_option("","--ohlc",action="store_true",dest="ohlcTF",default=False,
		help="plot stock OHLC Candlestick")
	parser.add_option("","--ohlc_combo",action="store_true",dest="ohlcComboTF",default=False,
		help="plot stock OHLC Candlestick + MA/RSI/MACD Combo")
	parser.add_option("","--src",action="store",dest="src",default="yh",
		help="data source (FILENAME is used if 'FILENAME' is provided. default: yh)")
	parser.add_option("","--start",action="store",dest="start",
		help="start YYYY-MM-DD, must pair with SRC (default: 2-years-ago)")
	parser.add_option("","--end",action="store",dest="end",
		help="end YYYY-MM-DD, must pair with SRC (default: today)")
	parser.add_option("","--days",action="store",dest="days",default=730,type=int,
		help="number of days from END date, must pair with SRC (default: 730)")
	parser.add_option("","--trendline",action="store_true",dest="trendTF",default=False,
		help="Draw trend-line, apply to the 1st array ONLY")
	parser.add_option("","--hiloline",action="store_true",dest="hiLoTF",default=False,
		help="Draw hilo-line, apply to the 1st array ONLY")
	parser.add_option("","--npar",action="store",dest="npar",default=15,type="int",
		help="trendline fitting polynomial degree (default: 15)")
	parser.add_option("","--j2ts",action="store",dest="j2ts",
		help="template script. Note, JINJA2 ONLY")
	parser.add_option("","--extra_js",action="store",dest="extraJS",
		help="extra JSON in DICT format.")
	parser.add_option("","--extra_xs",action="store",dest="extraXS",
		help="extra excutable string in k1=v1;k2=v2; format")
	parser.add_option("","--debug",action="store_true",dest="debugTF",default=False,
		help="debugging (default: False)")
	(options, args) = parser.parse_args(argv[1:])
	if retParser is True:
		return parser
	try:
		opts = vars(options)
		from _alan_str import extra_opts
		extra_opts(opts,xkey='extraJS',method='JS',updTF=True)
		extra_opts(opts,xkey='extraXS',method='XS',updTF=True)
	except Exception as e:
		sys.stderr.write("**ERROR:{} @{}\n".format(str(e),'opt_csv2plot'))
	return (opts, args)

def tst_csv2plotj2ts(args,opts=None,**optx):
	try:
		ret = run_csv2plot(args,opts,**optx)
		return ret
	except Exception as e:
		sys.stderr.write("**ERROR:{} @{}\n".format(str(e),'run_csv2plot'))
		return None, ''

if __name__ == '__main__':
	opts, args = opt_csv2plot(sys.argv)
	if 'run' in opts and opts['run'] in globals():
		ret = globals()[opts['run']](args,opts=opts)
	else:
		ret = get_csv2plot(args,opts)
		#ret = tst_csv2plotj2ts(args,opts)
