#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage of:
python3 -c "from plot_templates import plot_ts as pts;pts('^GSPC', chartname='',backend='tkAgg',debugTF=True)"
"""

import math
import sys
import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use('agg')
if matplotlib.get_backend().lower() != 'tkagg':
	matplotlib.use('Agg', force=True)
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import StrMethodFormatter
import matplotlib.dates as mdates
from datetime import datetime, timedelta, time
from _alan_calc import getKeyVal
from _alan_date import next_date,dt2ymd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm
font_path = "/usr/share/fonts/truetype/arphic/uming.ttc"

def fontProp(fname = "/usr/share/fonts/truetype/arphic/uming.ttc",**optx):
	return mfm.FontProperties(fname=font_path,**optx)

prop = fontProp()

#plt.style.use('classic')
if sys.version_info.major == 2:
	reload(sys)
	sys.setdefaultencoding('utf8')

def gen_chartpath(chartname='',chartdir='US/mp3_hourly',chartformat='svg',epoch=None,**optx):
	"""
	Generate 'chartpath', 'chartname' as
	  {chartdir}/{templateName}_{ticker}_{epoch}.{chartformat},
	  {templateName}_{ticker}_{epoch}.{chartformat}
	where
	  'chartdir','chartformat' and 'epoch' are optional
	Example,
	  gen_chartpath(templateName='stock_performance',ticker='AAPL')
	Note,
	  'ticker' with '^' or '=' characters will be removed
	"""
	chartname='' if not isinstance(chartname,str) else chartname
	if len(chartname)>0:
		if len(chartdir)>0:
			chartpath="{}/{}".format(chartdir,chartname)
		else:
			chartpath = chartname
		return chartpath,chartname
	if epoch is None:
		epoch = dt2ymd(next_date(),"%s")
	chartname = chartpath = ''
	ky=['templateName','ticker']
	kLst=getKeyVal(optx,ky,[],extendTF=True)
	kLst.extend([epoch,chartformat])
	if not any([ x is None for x in kLst]):
		chartname = "{}_{}_{}.{}".format(*kLst)
		chartname = re.sub('[\^=]','',chartname)
	if len(chartdir)>0 and len(chartname)>0:
		chartpath = "{}/{}".format(chartdir,chartname)
	else:
		chartpath = chartname
	return chartpath,chartname

def get_data(ticker,dv=[],src='yh',**optx):
	from _alan_calc import pullStockHistory as psd
	ranged = getKeyVal(optx,'ranged','1d')
	gap = getKeyVal(optx,'gap','1m')
	datax = psd(ticker,gap=gap,ranged=ranged,days=1500,src=src)
	df = datax
	print("===>",type(df),df.tail())
	if 'volume' in datax:
		dv = datax[['volume']]
	return (df,dv)

def plot_daily_conclusion(da,db=[],fig=None,ax=None,figsize=(11,6),backend='Agg',title='',chartformat='svg',debugTF=False,**optx):
	'''
	plot daily_peers by marketcap barh
	'''
	sys.stderr.write("===start:{}()\n".format("plot_daily_conclusion"))
	ticker = getKeyVal(optx,'ticker')
	if backend is not None:
		plt.switch_backend(backend)
	chartpath,chartname = gen_chartpath(**optx)
	sns.set(style="whitegrid")

	# define plot
	if fig is None:
		fig, ax = plt.subplots(figsize=figsize)
	ax.grid(False, axis='y')
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')

	bottom=optx['bottom'] if 'bottom' in optx else 0.25

	#da.plot(ax=ax,kind=kind,position=position,width=width)
	da = da[['comment','pchg']].sort_values(by='pchg',ascending=False).iloc[0:10,:]
	plt.title('指標變化百分比',fontproperties=fontProp(size=15))
	da['comment'] = [ xstr.strip('[,.，。]') for xstr in da['comment'].values ]
	ax.barh(y=da['comment'], width=da['pchg'])
	ax.set_yticklabels(labels=da['comment'],fontproperties=fontProp(size=15,weight='bold'))
	# plot features
	loc=optx['loc'] if 'loc' in optx else 9
	ncol=optx['ncol'] if 'ncol' in optx else 2
	fancybox=optx['fancybox'] if 'fancybox' in optx else True
	borderaxespad=optx['borderaxespad'] if 'borderaxespad' in optx else 0.
	plt.legend(loc=loc, ncol=ncol, fancybox=fancybox, borderaxespad=borderaxespad)
	rotation=optx['rotation'] if 'rotation' in optx else '30'
	fontsize=optx['fontsize'] if 'fontsize' in optx else 12
	plt.xticks(rotation=rotation,fontsize=fontsize)
	plt.legend(frameon=False)
	if len(title)>0:
		plt.title(title)
	plt.subplots_adjust(left=0.65,bottom=bottom)
	if chartname is not None and len(chartname)>5:
		plt.savefig(chartpath, format=chartformat)
	elif backend is not None and backend.lower()=='tkagg': 
		plt.show()
	return chartpath,ax,fig

def plot_mktidx(tkLst=[],fig=None,ax=None,figsize=(11,6),backend='Agg',title='',chartformat='svg',debugTF=False,**optx):
	sys.stderr.write("===start:{}()\n".format("plot_mktidx"))
	chartpath = getKeyVal(optx,'chartpath','')
	optx=locals()
	from csv2plotj2ts import get_csv2plot
	optx.update(optx['optx'])
	optx.pop('optx',None)
	opts ={'debugTF': True,'pngname': None, 'debugTF': True, 'rsiYN': True, 'renColumns': {'^IXIC': '納斯達克指數', '^DJI': '道瓊指數', '^GSPC': '標普500'}, 'colLst': 'close,pchg,epochs,ticker', 'hdrLst': None, 'npar': 15, 'days': 730, 'j2ts': None, 'start': None, 'pivot_group': 'ticker', 'gap': '1m', 'trendTF': False, 'ohlcTF': False, 'src': 'yh', 'pivot_value': 'close', 'end': None, 'title': '美股大盤走勢2019-09-17', 'lang': 'cn', 'backend': 'tkAgg', 'nbins': 6, 'tsTF': True, 'ranged': '1d', 'ohlcComboTF': False, 'sep': '|', 'interpolateYN': False, 'xaxis': 'epochs', 'pltStyle': 'classic', 'x_fmt': '%H:%M'}
	if len(tkLst)<1:
		tkLst= ['^GSPC', '^DJI', '^IXIC']
	plt.switch_backend(backend)
	if len(chartpath)>5:
		optx.update(pngname=chartpath)
	datax,fig,ax = get_csv2plot(tkLst,opts=opts,**optx)
	return(chartpath,ax,fig)

def plot_intraday_headline(tkLst=[],fig=None,ax=None,figsize=(11,6),backend='Agg',title='',chartformat='svg',debugTF=False,**optx):
	if backend is not None and len(backend)>1:
		plt.switch_backend(backend)
	sys.stderr.write("===start:{}()\n".format("plot_intraday_headline"))
	bottom=optx['bottom'] if 'bottom' in optx else 0.2
	chartpath = getKeyVal(optx,'chartpath','')
	chartname = getKeyVal(optx,'chartname','')    
	x_fmt = getKeyVal(optx,'x_fmt','%H:%M')    
	#chartpath,chartname = gen_chartpath(**optx)    
	optx=locals()
	from csv2plotj2ts import get_csv2plot,get_csvdata,subDict
	optx.update(optx['optx'])
	optx.pop('optx',None)
	opts ={'debugTF': True,'pngname': None, 'debugTF': True, 'rsiYN': True, 'renColumns': {'^IXIC': '納斯達克指數', '^DJI': '道瓊指數', '^GSPC': '標普500'}, 'colLst': 'change,pchg,close,epochs,ticker', 'hdrLst': None, 'npar': 15, 'days': 730, 'j2ts': None, 'start': None, 'pivot_group': 'ticker', 'gap': '1m', 'trendTF': False, 'ohlcTF': False, 'src': 'yh', 'pivot_value': 'close', 'pchgTF':True, 'end': None, 'title': '美股大盤走勢2019-09-17', 'lang': 'cn', 'backend': 'tkAgg', 'nbins': 6, 'tsTF': True, 'ranged': '1d', 'ohlcComboTF': False, 'sep': '|', 'interpolateYN': False, 'xaxis': 'epochs', 'pltStyle': 'classic', 'x_fmt': '%H:%M'}
	if len(tkLst)<1:
		tkLst= ['^GSPC', '^DJI', '^IXIC']
	if len(chartpath)>5:
		optx.update(pngname=chartpath)
	renColumns = opts['renColumns']
	gkys=['sep','src','days','start','end','colLst','hdrLst','debugTF','ranged','gap','tsTF','pchgTF','searchDB']
	df = get_csvdata(tkLst, **subDict(opts,gkys))
	if df is None or len(df)<1 or df.size<1:
		return '',{},{}
	sys.stderr.write(" --opts:{},DF:\n{}".format(subDict(opts,gkys),df.tail()) )    
	
	da=pd.DataFrame()
	dc=pd.DataFrame()
	for tk in tkLst:
		da[tk] = df[df['ticker']==tk]['pchg']    
		dc[tk] = df[df['ticker']==tk]['change']    
	#da = (da.pct_change().fillna(0)+1).cumprod()   
    
	if debugTF:
		sys.stderr.write("\n===== da:\n{}\n\t@{}\n".format(da.to_csv(),'plot_intraday_headline'))
	
	max_num = da.values.max()
	min_num = da.values.min()
	min_index = np.min(da.index)
	x_lim = min_index.replace(hour=16, minute =0 )    

	try:
		plt.rcParams.update(plt.rcParamsDefault)            
		sns.set_style("whitegrid",{'grid.linestyle':'--'})                 
		fig, ax=plt.subplots(figsize=figsize)    
		fig.autofmt_xdate()

		plt.xlim(min_index-timedelta(minutes=5), x_lim+timedelta(minutes=5))
		##plt.ylim(min_num-(max_num-min_num)/2, max_num+(max_num-min_num)/5)
		#plt.xlabel('Time',fontsize=15)
		#plt.ylabel('price',fontsize=15)
		#hours = mdates.HourLocator(interval = 1)
		h_fmt = mdates.DateFormatter(x_fmt)
		#ax.xaxis.set_major_locator(hours)
		ax.xaxis.set_major_formatter(h_fmt)
		ax.grid(False, axis='x')                        
		ax.spines['right'].set_color('none')
		ax.spines['top'].set_color('none')
		ax.spines['bottom'].set_color('none')
		ax.spines['left'].set_color('none')
		#ax.set_ylabel('Returns Since Previous Close in %')
		ax.set_ylabel('自前日收盤價報酬率%',fontproperties=fontProp())
		plt.title(title,fontproperties=fontProp(size=15))
		#laLst=('標普500','道瓊指數','納斯達克指數')
		#for j,xlabel in enumerate(laLst):
		for j,tkX in enumerate(tkLst):
			xlabel = renColumns[tkX]
			xlev=df.loc[df['ticker']==tkX,'close'].dropna().iloc[-1] 
			xchg=da[tkLst[j]].dropna().iloc[-1]
			xchange=dc[tkLst[j]].dropna().iloc[-1]
			xa = "{:s}: {:,.0f} ({:+,.0f}, {:,.2%})".format(xlabel,xlev,xchange,xchg)
			plt.plot(da.index, da[tkX],label=xa)
        
		ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: '{:,.2%}'.format(x)))
		plt.legend(prop=fontProp())        
		if debugTF:
			sys.stderr.write("===plot data:\n {}\n".format(df.tail(2)))
			sys.stderr.write("===plot pchg:\n {}\n".format(da.tail(2)))
		plt.subplots_adjust(left=0.1,bottom=bottom)
		if chartname is not None and len(chartname)>5:
			plt.savefig(chartpath, format=chartformat) #, bbox_inches='tight',dpi=1000)
		elif backend is not None and backend.lower()=='tkagg': 
			plt.show()
	except Exception as e:
		sys.stderr.write("**ERROR: {} @{}\n".format(str(e),"plot_intraday_headline"))
		return '',None,None   
	return(chartpath,ax,fig)


def plot_daily_macro(df,dv=[],fig=None,ax=None,figsize=(11,6),backend='Agg',title='',chartformat='svg',debugTF=True,**optx):
	if len(df)<1:
		return '',None,None
	sys.stderr.write("===start:{}()\n".format("plot_daily_macro"))
	src = getKeyVal(optx,'src','fred')
	if not isinstance(df,pd.DataFrame):
		ticker=df
		from _alan_calc import pullStockHistory as psd
		ranged = getKeyVal(optx,'ranged','1d')
		gap = getKeyVal(optx,'gap','1m')
		datax = psd(ticker,gap=gap,ranged=ranged,days=1500,src=src)
		datax.rename(columns={'close':ticker}, inplace=True)
		df = datax[[ticker]]
		if src=='fred':
			tk2=ticker+'_PCTCHG'
			dx2 = psd(tk2,src=src,days=1500)
			if len(dx2)>0:
				dv = datax[[ticker]]
				df = dx2[['close']]
		elif 'volume' in datax:
			dv = datax[['volume']]
	if 'volume' in df:
		dv = df[['volume']].copy()
		cs = list(df.columns)
		cs.pop(cs.index('volume'))
		df = df[cs]
	if backend is not None and len(backend)>1:
		plt.switch_backend(backend)
	trendTF = getKeyVal(optx,'trendTF',True)
	bottom=optx['bottom'] if 'bottom' in optx else 0.35
	chartpath,chartname = gen_chartpath(**optx)
	try:
		from _alan_pppscf import vertex_locator
		if not isinstance(df,pd.DataFrame):
			df = pd.DataFrame(data=df)
		if fig is None:
			plt.rcParams.update(plt.rcParamsDefault)            
			sns.set_style("whitegrid",{'grid.linestyle':'--'})                 
			fig, ax=plt.subplots(figsize=figsize)
			ax.grid(False, axis='x')                        
			ax.spines['right'].set_color('none')
			ax.spines['top'].set_color('none')
			ax.spines['bottom'].set_color('none')
			ax.spines['left'].set_color('none')            
		if debugTF:
			sys.stderr.write("===== DF:\n{}\n".format(df.tail()))
		#ax.plot(df.index,df[ticker])
		#df.plot(ax=ax)
		ax.plot(df.index,df.iloc[:,0])
		x_fmt = getKeyVal(optx,'x_fmt','') 
		yunit=['','1,000','Million','Billion','Trillion']
		ymax = df.iloc[:,0].max()
		ndg = max(int(math.log(ymax,10)/3),1)
		nadj = (ndg-1)*3
		ylabel = yunit[ndg-1] or ''
		if debugTF:
			sys.stderr.write("===== {} {} {} {}\n".format(ymax,ndg,nadj,yunit[ndg-1]))
		if ndg>1:
			ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: '{:,g}'.format(x/10**nadj)))
		else:
			ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: '{:,g}'.format(x)))
		ax.set_ylabel(ylabel)
		if trendTF is True:
			npar = getKeyVal(optx,'npar',6)
			trendName = df._get_numeric_data().columns[0]
			dy =  df[trendName]
			dg, dh = vertex_locator(dy,npar=npar,debugTF=debugTF)
			ax.plot(dg.index,dg.iloc[:,0])
			if debugTF:
				sys.stderr.write("===plot data:\n {}\n".format(dg.tail()))
		if len(dv)>0 and isinstance(dv,pd.DataFrame):
			axv = ax.twinx()
			kind2='area' if src!='fred' else 'line'
			color2='lightgray' if src!='fred' else 'gray'
			axv.grid(False, axis='both')            
			axv.spines['right'].set_color('none')
			axv.spines['top'].set_color('none')
			axv.spines['bottom'].set_color('none')
			axv.spines['left'].set_color('none') 
			axv.plot(dv.index,dv.iloc[:,0],color=color2,alpha=.4)
			y2max = dv.iloc[:,0].max()
			ndg = max(int(math.log(y2max,10)/3),1)
			nadj = (ndg-1)*3
			if ndg>1:
				axv.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: '{:,g}'.format(x/10**nadj)))
			else:
				axv.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: '{:,g}'.format(x)))
			ylabel2 = yunit[ndg-1] or ''
			axv.set_ylabel(ylabel2)
			if debugTF:
				sys.stderr.write("===plot data:\n {}\n".format(dv.tail()))
		if debugTF:
			sys.stderr.write("===plot data:\n {}\n".format(df.tail(20)))
		plt.subplots_adjust(left=0.1,bottom=bottom)
		plt.title(title,fontproperties=fontProp())
		if len(x_fmt)>0: # TBD, specify xticks format
			ax.xaxis.set_major_formatter(mdates.DateFormatter(x_fmt))
		ax.grid(linestyle='dotted',linewidth=0.5)
		if chartname is not None and len(chartname)>5:
			plt.savefig(chartpath, format=chartformat) #, bbox_inches='tight',dpi=1000)
		elif backend is not None and backend.lower()=='tkagg': 
			plt.show()
	except Exception as e:
		sys.stderr.write("**ERROR: {} of {} @{}\n".format(str(e),chartpath,"plot_daily_macro"))
		return '',None,None
	return(chartpath,ax,fig)

def plot_daily_mostactive(da,db=[],fig=None,ax=None,figsize=(11,6),backend='Agg',title='',chartformat='svg',debugTF=False,**optx):
	'''
	plot daily_peers by marketcap barh
	'''
	sys.stderr.write("===start:{}()\n".format("plot_daily_mostactive"))
	ticker = getKeyVal(optx,'ticker')
	if backend is not None:
		plt.switch_backend(backend)
	chartpath,chartname = gen_chartpath(**optx)
	sns.set(style="whitegrid")

	# define plot
	if fig is None:
		fig, ax = plt.subplots(figsize=figsize)
	ax.grid(False, axis='y')
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')

	bottom=optx['bottom'] if 'bottom' in optx else 0.3

	#da.plot(ax=ax,kind=kind,position=position,width=width)

	#xmax = da['changePercent'].max()
	#ndg = max(int(math.log(xmax,10)/3),1)
	#nadj = (ndg-1)*3
	#yunit=['','1,000','Million','Billion','Trillion']
	#ylabel = yunit[ndg-1] or ''
	#ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: '{:,g}'.format(x/10**nadj)))
	da = da.sort_values(by='changePercent')        
	tmp = ax.barh(y=da['ticker'], width=da['changePercent'])
    
	da['pos'] = range(0,len(da))         
	#pos=int(dh[dh.index==ticker]['pos'])
	#axes.patches[pos].set_facecolor('#aa3333')        
	for sbar in tmp:
		height = sbar.get_width()            
		ax.annotate('{}'.format(round(height,2)),xy=(height, sbar.get_y()),xytext=((3 if height>0 else -3),3),textcoords="offset points",ha=('left' if height>0 else 'right'), va='bottom')
	#plt.legend(frameon=False) 

	plt.xlim((da['changePercent'].min()*1.3 if da['changePercent'].min()<0 else 0) , right = da['changePercent'].max()*1.2)
        
	plt.title('changePercent')    
	#barchart = sns.barplot(x="changePercent", y='ticker', data=da,palette="Blues_d")
	#barchart.set(ylabel='ticker')#xlabel=ylabel)
	#pos=int(da[da['ticker']==ticker].index.values)
	#ax.patches[pos].set_facecolor('#aa3333')
	# plot features
	loc=optx['loc'] if 'loc' in optx else 9
	ncol=optx['ncol'] if 'ncol' in optx else 2
	fancybox=optx['fancybox'] if 'fancybox' in optx else True
	borderaxespad=optx['borderaxespad'] if 'borderaxespad' in optx else 0.
	#plt.legend(loc=loc, ncol=ncol, fancybox=fancybox, borderaxespad=borderaxespad)
	rotation=optx['rotation'] if 'rotation' in optx else '30'
	fontsize=optx['fontsize'] if 'fontsize' in optx else 12
	plt.xticks(rotation=rotation,fontsize=fontsize)
	plt.legend(frameon=False)
	plt.title('changePercent')    
	if len(title)>0:
		plt.title(title)
	plt.subplots_adjust(left=0.1,bottom=bottom)
	if chartname is not None and len(chartname)>5:
		plt.savefig(chartpath, format=chartformat)
	elif backend is not None and backend.lower()=='tkagg': 
		plt.show()
	return chartpath,ax,fig

def plot_stock_performance(df,fig=None,ax=None,figsize=(11,6),backend='Agg',title='',chartformat='svg',debugTF=True,**optx):
	sys.stderr.write("===start:{}()\n".format("plot_stock_performance"))
	if isinstance(df,str):
		df,dv = get_data(df,**optx)
	elif not isinstance(df,pd.DataFrame):
		df = pd.DataFrame(data=df)
	if len(df)<1:
		return '',None,None
	if backend is not None and len(backend)>1:
		plt.switch_backend(backend)
	bottom=optx['bottom'] if 'bottom' in optx else 0.2
	chartpath,chartname = gen_chartpath(**optx)
	trendTF = getKeyVal(optx,'trendTF',True)    
	x_fmt = getKeyVal(optx,'x_fmt','%H:%M')    
	max_num = df['close'].max()
	max_num_date = df[df['close'] == max_num].iloc[-1].name
	min_num = np.min(df['close'])
	min_num_date = df[df['close'] == min_num].iloc[-1].name
	curr_num = df['close'][-1]
	curr_num_date = df[df['close'] == curr_num].iloc[-1].name
	open_num = df['close'].iloc[0]    
	max_index = np.max(df.index)
	min_index = np.min(df.index)
	x_lim = min_index.replace(hour=16, minute =31 )
	xprice = getKeyVal(optx,'xprice',open_num)
	cpchg = getKeyVal(optx,'cpchg',round((curr_num/open_num-1)*100 ,2)) 
	max_ylim = np.max([max_num,xprice])
	min_ylim = np.min([min_num,xprice])   
	try:
		from _alan_pppscf import vertex_locator
		if trendTF is True:
			try:
				npar = getKeyVal(optx,'npar',6)
				trendName = df._get_numeric_data().columns[0]
				dy = df[trendName]
				dg, dh = vertex_locator(dy,npar=npar,debugTF=debugTF)
				dh = dg
			except Exception as e:
				sys.stderr.write("**ERROR:{} @ {}\n".format(str(e),"running vertex_locator()"))
				dh={}
			#print(dh)            
		else:
			dh={}
		if not isinstance(df,pd.DataFrame):
			df = pd.DataFrame(data=df)
		if fig is None:
			plt.rcParams.update(plt.rcParamsDefault)            
			sns.set_style("whitegrid",{'grid.linestyle':'--'})                 
			fig, ax=plt.subplots(figsize=figsize)
			fig.autofmt_xdate()
			ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
			plt.xlim(min_index-timedelta(minutes=5), x_lim+timedelta(minutes=5))
			plt.ylim(min_ylim-(max_ylim-min_ylim)/2, max_ylim+(max_ylim-min_ylim)/5)
			plt.subplots_adjust(bottom=0.2)
			#plt.xlabel('Time',fontsize=15)
			#plt.ylabel('price',fontsize=15)
			#hours = mdates.HourLocator(interval = 1)
			h_fmt = mdates.DateFormatter(x_fmt)
			#ax.xaxis.set_major_locator(hours)
			ax.xaxis.set_major_formatter(h_fmt)
			ax.grid(False, axis='x')                        
			ax.spines['right'].set_color('none')
			ax.spines['top'].set_color('none')
			ax.spines['bottom'].set_color('none')
			ax.spines['left'].set_color('none')
			plt.title(title,fontproperties=fontProp(size=15))
			plt.legend(frameon=False)            
		plt.plot(df.index, df['close'])
		plt.plot([min_index,x_lim],[xprice,xprice],'k--',lw=1)# plot open dash line
		plt.plot([curr_num_date,curr_num_date],[xprice,curr_num],'k--',lw=1)# plot day pchg vertical line
		#plt.annotate('{}%'.format(round((curr_num/open_num-1)*100,2)), xy=(curr_num_date,open_num+(curr_num-open_num)/2),xytext=(0,6),fontsize=10,annotation_clip=True, textcoords="offset points",ha=('left'), va='center') # plot pchg number        
		plt.scatter(max_num_date,max_num,s=20,color = 'g',zorder = 19)#plot max point
		plt.scatter(min_num_date,min_num,s=20,color = 'r',zorder = 18)#plot min point
		plt.scatter(curr_num_date,curr_num,s=40,zorder = 20)#plot curr point
		plt.annotate('{}'.format(round(max_num,2)), xy=(max_num_date,max_num),xytext=(0,20),fontsize=10,arrowprops=dict(arrowstyle='simple'),annotation_clip=True, textcoords="offset points",ha=('right'), va='bottom') # plot max point number
		plt.annotate('{}'.format(round(min_num,2)), xy=(min_num_date,min_num),xytext=(0,-20),fontsize=10,arrowprops=dict(arrowstyle='simple'),annotation_clip=True, textcoords="offset points",ha=('left'), va='top') # plot min point number
		plt.annotate('{} | {}%'.format(round(curr_num,2),round(cpchg,2)), xy=(curr_num_date,curr_num),xytext=(0,20),fontsize=10,arrowprops=dict(arrowstyle='simple'),annotation_clip=True, textcoords="offset points",ha=('left'), va='bottom') # plot curr point number
		if len(dh)>1:
			if dh.iloc[-1].values<dh.iloc[-2].values:
				ar_color = 'red'
			else:
				ar_color= 'green'        
			plt.annotate('',xy=(dh.iloc[-1].name,dh.iloc[-1]*1.00),xytext=(dh.iloc[-2].name,dh.iloc[-2]*1.00), arrowprops=dict(facecolor=ar_color, shrink=0.05),annotation_clip=True) # plot trend arrow
			plt.plot(dh.index, dh.values)
		if 'volume' in df:                       
			axv = ax.twinx()
			axv.grid(False, axis='both')
			axv.spines['right'].set_color('none')
			axv.spines['top'].set_color('none')
			axv.spines['bottom'].set_color('none')
			axv.spines['left'].set_color('none')
			axv.xaxis.set_major_formatter(h_fmt)
			axv.set_yticklabels([]) 
			axv.tick_params(right="off")      
			plt.ylim(0,df['volume'].max()*3.5)            
			#axv.plot(df['volume'].index, df['volume'], color='gray',alpha=0.8)
			axv.fill_between(df['volume'].index,df['volume'], alpha =0.8)             
		if debugTF:
			sys.stderr.write("===plot data:\n {}\n".format(df.tail()))
		plt.subplots_adjust(left=0.1,bottom=bottom)
		if chartname is not None and len(chartname)>5:
			plt.savefig(chartpath, format=chartformat) #, bbox_inches='tight',dpi=1000)
		elif backend is not None and backend.lower()=='tkagg': 
			plt.show()
	except Exception as e:
		sys.stderr.write("**ERROR: {} @{}\n".format(str(e),"plot_stock_performance"))
		return '',None,None
	return(chartpath,ax,fig)


def plot_daily_peers(da,db=[],fig=None,ax=None,figsize=(11,6),backend='Agg',title='',chartformat='svg',debugTF=False,**optx):
	'''
	plot daily_peers by marketcap barh
	'''
	sys.stderr.write("===start:{}()\n".format("plot_daily_peers"))
	if not isinstance(da,pd.DataFrame) or len(da)<1:
		return {},{},{}   
	ticker = getKeyVal(optx,'ticker')
	if backend is not None:
		plt.switch_backend(backend)
	chartpath,chartname = gen_chartpath(**optx)
	sns.set(style="whitegrid")

	# define plot
	if fig is None:
		fig, ax = plt.subplots(figsize=figsize)
	ax.grid(False, axis='y')
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')

	bottom=optx['bottom'] if 'bottom' in optx else 0.3

	#da.plot(ax=ax,kind=kind,position=position,width=width)

	plt.xlim(0,da['marketcap'].max()*1.2)
	plt.title('Marketcap')

	xmax = da['marketcap'].max()
	ndg = max(int(math.log(xmax,10)/3),1)
	nadj = (ndg-1)*3
	yunit=['','1,000','Million','Billion','Trillion']
	ylabel = yunit[ndg-1] or ''
	ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos: '{:,g}'.format(x/10**nadj)))

	barchart = sns.barplot(x="marketcap", y='peers', data=da,palette="Blues_d")
	barchart.set(xlabel=ylabel, ylabel='Peers')
	pos=int(da[da['peers']==ticker].index.values)
	ax.patches[pos].set_facecolor('#aa3333')
	# plot features
	loc=optx['loc'] if 'loc' in optx else 0
	ncol=optx['ncol'] if 'ncol' in optx else 0
	#if debugTF:
	sys.stderr.write(" --xmax,ndg,nadj,yuni: {} {} {} {}\n".format(xmax,ndg,nadj,yunit[ndg-1]))
	sys.stderr.write(" --optx:{}\n --loc:{},ncol:{}:data:\n{}\n".format(optx,loc,ncol,da))
	fancybox=optx['fancybox'] if 'fancybox' in optx else True
	borderaxespad=optx['borderaxespad'] if 'borderaxespad' in optx else 0.
	plt.legend(loc=loc, ncol=ncol, fancybox=fancybox, borderaxespad=borderaxespad)
	rotation=optx['rotation'] if 'rotation' in optx else '30'
	fontsize=optx['fontsize'] if 'fontsize' in optx else 12
	plt.xticks(rotation=rotation,fontsize=fontsize)
	plt.legend(frameon=False)
	if len(title)>0:
		plt.title(title)
	plt.subplots_adjust(left=0.1,bottom=bottom)
	if chartname is not None and len(chartname)>5:
		plt.savefig(chartpath, format=chartformat)
	elif backend is not None and backend.lower()=='tkagg': 
		plt.show()
	return chartpath,ax,fig

def plot_peers_performance(df,fig=None,ax=None,figsize=(11,6),backend='Agg',title='',chartformat='svg',debugTF=False,**optx):
	sns.set(style="whitegrid")
	df['pchg']=df['pchg']*100
	if 'ticker' in df and 'peRatio' in df:
		df = pd.DataFrame(df.sort_values(by='peRatio').set_index('ticker',drop=True))
	elif 'ticker' in df:
		df = pd.DataFrame(df.set_index('ticker',drop=True))
	ngrid = df.shape[1]
	kind = getKeyVal(optx,'kind','barh')
	pctTF = getKeyVal(optx,'pctTF',False)
	vtitle = getKeyVal(optx,'vtitle',["1-Day Price Change %","PE Ratio"])
	ticker = getKeyVal(optx,'ticker')    
	if pctTF is True:
		vfmt = getKeyVal(optx,'vfmt',["{:.2}%","{:.0f}"])
	else:
		vfmt = getKeyVal(optx,'vfmt',["{:.2%}","{:.0f}"])
	if len(df)<1:
		return '',None,None
	if backend is not None and len(backend)>1:
		plt.switch_backend(backend)
	bottom=optx['bottom'] if 'bottom' in optx else 0.3
	#style = getKeyVal(optx,'style','classic')
	#plt.style.use(style)
	chartpath,chartname = gen_chartpath(**optx)
	if ngrid>1: 
		if np.isnan(df.loc[ticker,'peRatio']):
			peTF = False
		else:
			peTF = True
	#print(peTF)
	if peTF is True:
		ncols = 2
		fig, axes = plt.subplots(nrows=1, ncols=ncols,figsize=figsize)
		for i in range(ncols):
			axes[i].spines['right'].set_color('none')
			axes[i].spines['top'].set_color('none')
			axes[i].spines['bottom'].set_color('none')
			axes[i].spines['left'].set_color('none')
			axes[i].set_title(vtitle[i])          
			axes[i].grid(False,axis= 'y')
			field = 'pchg' if i == 0 else 'peRatio'            
			dh = df.sort_values(by=field)
			plt.xlim((dh[field].min()*1.3 if dh[field].min()<0 else 0) , right = dh[field].max()*1.2)           
			tmp=axes[i].barh(y=dh.index.values, width=dh[field])
			dh['pos'] = range(0,len(dh))         
			pos=int(dh[dh.index==ticker]['pos'])
			axes[i].patches[pos].set_facecolor('#aa3333')      
			for sbar in tmp:
				height = sbar.get_width()            
				axes[i].annotate('{}'.format(round(height,2)),xy=(height, sbar.get_y()),xytext=((3 if height>0 else -3),3),textcoords="offset points",ha=('left' if height>0 else 'right'), va='bottom')
                                                                                             
	else:
		ncols = 1
		fig, axes = plt.subplots(nrows=1, ncols=ncols,figsize=figsize)
		axes.spines['right'].set_color('none')
		axes.spines['top'].set_color('none')
		axes.spines['bottom'].set_color('none')
		axes.spines['left'].set_color('none')
		axes.set_title(vtitle[0])    
		axes.grid(False, axis='y')
		dh = df.sort_values(by='pchg')        
		tmp = axes.barh(y=dh.index.values, width=dh['pchg'])
		dh['pos'] = range(0,len(dh))         
		pos=int(dh[dh.index==ticker]['pos'])
		axes.patches[pos].set_facecolor('#aa3333')        
		for sbar in tmp:
			height = sbar.get_width()            
			axes.annotate('{}'.format(round(height,2)),xy=(height, sbar.get_y()),xytext=((3 if height>0 else -3),3),textcoords="offset points",ha=('left' if height>0 else 'right'), va='bottom')
	plt.legend(frameon=False)       
	#axes = df.plot(kind=kind,legend=False,ax=plt.gca(),subplots=True,sharey=True)
	#for j in range(ngrid):
		#if vtitle[j] is not None and len(vtitle[j])>0:
			#ax0.set_title(vtitle[j])
		#for i, v in enumerate(df.iloc[:,j]):
			#ax1.text(v , i + .05, vfmt[j].format(v), fontweight='bold')
	plt.subplots_adjust(left=.14, bottom=bottom, right=.90, top=.90, wspace=.20, hspace=.50)
	if chartname is not None and len(chartname)>5:
		plt.savefig(chartpath, format=chartformat) #, bbox_inches='tight',dpi=1000)
	elif backend is not None and backend.lower()=='tkagg': 
		plt.show()
	return(chartpath,axes,fig)

def plot_earnings_performance(da,db=[],fig=None,ax=None,figsize=(11,6),backend='Agg',title='',chartformat='svg',debugTF=False,**optx):
	'''
	plot daily_peers by marketcap barh
	'''
	ticker = getKeyVal(optx,'ticker')
	if backend is not None:
		plt.switch_backend(backend)
	chartpath,chartname = gen_chartpath(**optx)
	sns.set(style="whitegrid")

	# define plot
	if fig is None:
		fig, ax = plt.subplots(figsize=figsize)
	ax.grid(False, axis='x')
	ax.spines['right'].set_color('none')
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')

	bottom=optx['bottom'] if 'bottom' in optx else 0.3
	da['pbdate'] = da.index.astype(str)
	da = da.sort_index(ascending=True)
	#da.index = pd.to_datetime(da.index)    
	#plt.ylim(,da['actualEPS'].max()*1.2)
	plt.title('EPS')

	xmax = abs(da['actualEPS']).max()
	ndg = max(int(math.log(xmax,10)/3),1)
	nadj = (ndg-1)*3
	yunit=['','1,000','Million','Billion','Trillion']
	ylabel = yunit[ndg-1] or ''
	if debugTF:
		sys.stderr.write(" --xmax:{},ndg:{},nadj:{},yunit:{}\n".format(xmax,ndg,nadj,yunit[ndg-1]))
	ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: '{:,g}'.format(x/10**nadj)))
	tmp = ax.bar(x=da['pbdate'].values, height=da['actualEPS'])
	#tmp = sns.barplot(x='pbdate',y='actualEPS',data=da,palette="Blues_d")
	#height = tmp.get_height()
	for rect in tmp:
		height = rect.get_height()
		ax.annotate('{}'.format(height),xy=(rect.get_x()+rect.get_width() / 2, height),xytext=(0, 3),textcoords="offset points",ha='center', va='bottom')    
	#sns.catplot('pbdate','actualEPS',data=da,kind='point')
	#barchart = sns.barplot(x='pbdate',y='actualEPS',data=da,palette="Blues_d")
	#barchart.set(xlabel=ylabel, ylabel='actualEPS')
	#for index, row in da.iterrows():
		#rint(row)        
		#barchart.text(row['pbdate'],row['actualEPS'], round(row['actualEPS'],2), color='black', ha="center")
	#sns.catplot('pbdate','actualEPS',data=da,kind='point')
	#print(da)
	pos=len(da)-1
	ax.patches[pos].set_facecolor('#aa3333')
	# plot features
	loc=optx['loc'] if 'loc' in optx else 9
	ncol=optx['ncol'] if 'ncol' in optx else 2
	fancybox=optx['fancybox'] if 'fancybox' in optx else True
	borderaxespad=optx['borderaxespad'] if 'borderaxespad' in optx else 0.
	plt.legend(loc=loc, ncol=ncol, fancybox=fancybox, borderaxespad=borderaxespad)
	rotation=optx['rotation'] if 'rotation' in optx else '30'
	fontsize=optx['fontsize'] if 'fontsize' in optx else 12
	plt.xticks(rotation=rotation,fontsize=fontsize)
	plt.legend(frameon=False)
	if len(title)>0:
		plt.title(title)
	plt.subplots_adjust(left=0.1,bottom=bottom)
	if chartname is not None and len(chartname)>5:
		plt.savefig(chartpath, format=chartformat)
	elif backend is not None and backend.lower()=='tkagg': 
		plt.show()
	return chartpath,ax,fig

def plot_ts(df,dv=[],fig=None,ax=None,figsize=(11,6),backend='Agg',title='',chartformat='svg',debugTF=False,**optx):
	if len(df)<1:
		return '',None,None
	sys.stderr.write("===start:{}()\n".format("plot_ts"))
	src = getKeyVal(optx,'src','yh')
	#if isinstance(df,str):
	if not isinstance(df,pd.DataFrame):
		ticker=df
		from _alan_calc import pullStockHistory as psd
		ranged = getKeyVal(optx,'ranged','1d')
		gap = getKeyVal(optx,'gap','1m')
		datax = psd(ticker,gap=gap,ranged=ranged,days=1500,src=src)
		datax.rename(columns={'close':ticker}, inplace=True)
		df = datax[[ticker]]
		if src=='fred':
			tk2=ticker+'_PCTCHG'
			dx2 = psd(tk2,src=src,days=1500)
			if len(dx2)>0:
				dv = datax[[ticker]]
				df = dx2[['close']]
		elif 'volume' in datax:
			dv = datax[['volume']]
	if 'volume' in df:
		dv = df[['volume']].copy()
		cs = list(df.columns)
		cs.pop(cs.index('volume'))
		df = df[cs]
	if backend is not None and len(backend)>1:
		plt.switch_backend(backend)
	trendTF = getKeyVal(optx,'trendTF',True)
	bottom=optx['bottom'] if 'bottom' in optx else 0.35
	chartpath,chartname = gen_chartpath(**optx)
	try:
		from _alan_pppscf import vertex_locator
		if not isinstance(df,pd.DataFrame):
			df = pd.DataFrame(data=df)
		if fig is None:
			sns.set_style("whitegrid",{'grid.linestyle':'--'})            
			fig, ax=plt.subplots(figsize=figsize)
			ax.grid(False, axis='x')
			ax.spines['right'].set_color('none')
			ax.spines['top'].set_color('none')
			ax.spines['bottom'].set_color('none')
			ax.spines['left'].set_color('none')            
		plt.plot(df.index,df.iloc[:,0])
		plt.title(title)        
		x_fmt = getKeyVal(optx,'x_fmt','') 
		yunit=['','1,000','Million','Billion','Trillion']
		ymax = abs(df.iloc[:,0]).max()
		ndg = max(int(math.log(ymax,10)/3),1)
		nadj = (ndg-1)*3
		ylabel = yunit[ndg-1] or ''
		if debugTF:
			sys.stderr.write("+++++ {} {} {} {}\n".format(ymax,ndg,nadj,yunit[ndg-1]))
		if ndg>1:
			ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: '{:,g}'.format(x/10**nadj)))
		else:
			ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: '{:,g}'.format(x)))
		ax.set_ylabel(ylabel)
		if trendTF is True:
			npar = getKeyVal(optx,'npar',6)
			trendName = df._get_numeric_data().columns[0]
			dy =  df[trendName]
			dg, dh = vertex_locator(dy,npar=npar,debugTF=debugTF)
			dg.plot(ax=ax,legend=False)
			if debugTF:
				sys.stderr.write("===plot data:\n {}\n".format(dg.tail()))
		if len(dv)>0 and isinstance(dv,pd.DataFrame):
			axv = ax.twinx()
			kind2='area' if src!='fred' else 'line'
			color2='lightgray' if src!='fred' else 'gray'
			axv = dv.plot(ax=axv,kind=kind2,color=color2,alpha=.4,legend=False)
			y2max = dv.iloc[:,0].max()
			ndg = max(int(math.log(y2max,10)/3),1)
			nadj = (ndg-1)*3
			if ndg>1:
				axv.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: '{:,g}'.format(x/10**nadj)))
			else:
				axv.yaxis.set_major_formatter(FuncFormatter(lambda x,pos: '{:,g}'.format(x)))
			ylabel2 = yunit[ndg-1] or ''
			axv.set_ylabel(ylabel2)
			if debugTF:
				sys.stderr.write("===plot data:\n {}\n".format(dv.tail()))
		if debugTF:
			sys.stderr.write("===plot data:\n {}\n".format(df.tail(20)))
		plt.subplots_adjust(left=0.1,bottom=bottom)
		#if len(x_fmt)>0: # TBD, specify xticks format
		#	ax.xaxis.set_major_formatter(mdates.DateFormatter(x_fmt))
		ax.grid(linestyle='dotted',linewidth=0.5)
		if chartname is not None and len(chartname)>5:
			plt.savefig(chartpath, format=chartformat) #, bbox_inches='tight',dpi=1000)
		elif backend is not None and backend.lower()=='tkagg': 
			plt.show()
	except Exception as e:
		sys.stderr.write("**ERROR: {} @{}\n".format(str(e),"plot_ts"))
		sys.stderr.write("{}\n".format(locals()))
		return '',None,None
	return(chartpath,ax,fig)

def plot_barh2chart(df,fig=None,ax=None,figsize=(11,6),backend='Agg',title='',chartformat='svg',debugTF=False,**optx):
	if 'ticker' in df and 'peRatio' in df:
		df = pd.DataFrame(df.dropna().sort_values(by='peRatio').set_index('ticker',drop=True))
	elif 'ticker' in df:
		df = pd.DataFrame(df.dropna().set_index('ticker',drop=True))
	ngrid = df.shape[1]
	kind = getKeyVal(optx,'kind','barh')
	pctTF = getKeyVal(optx,'pctTF',False)
	vtitle = getKeyVal(optx,'vtitle',["1-Day Price Change %","PE Ratio"])
	if pctTF is True:
		vfmt = getKeyVal(optx,'vfmt',["{:.2}%","{:.0f}"])
	else:
		vfmt = getKeyVal(optx,'vfmt',["{:.2%}","{:.0f}"])
	if len(df)<1:
		return '',None,None
	if backend is not None and len(backend)>1:
		plt.switch_backend(backend)
	bottom=optx['bottom'] if 'bottom' in optx else 0.35
	style = getKeyVal(optx,'style','classic')
	plt.style.use(style)
	chartpath,chartname = gen_chartpath(**optx)
	fig, axes = plt.subplots(ngrid,figsize=figsize,sharey=True)
	axes = df.plot(kind=kind,legend=False,ax=plt.gca(),subplots=True,sharey=True)
	for j in range(ngrid):
		if vtitle[j] is not None and len(vtitle[j])>0:
			axes[j].set_title(vtitle[j])
		for i, v in enumerate(df.iloc[:,j]):
			axes[j].text(v , i + .05, vfmt[j].format(v), fontweight='bold')
	plt.subplots_adjust(left=.1, bottom=bottom, right=.90, top=.90, wspace=.20, hspace=.50)
	if chartname is not None and len(chartname)>5:
		plt.savefig(chartpath, format=chartformat) #, bbox_inches='tight',dpi=1000)
	elif backend is not None and backend.lower()=='tkagg': 
		plt.show()
	return(chartpath,axes,fig)

def plot_2yaxis(da,db=[],fig=None,ax=None,figsize=(11,6),backend='Agg',title='',chartformat='svg',debugTF=False,**optx):
	'''
	plot 2-YAxis via da and db data
	'''
	if backend is not None:
		plt.switch_backend(backend)
	chartpath,chartname = gen_chartpath(**optx)
	# define plot
	if fig is None:
		fig, ax = plt.subplots(figsize=figsize)
	# 1st plot
	bottom=optx['bottom'] if 'bottom' in optx else 0.35
	kind=optx['kind'] if 'kind' in optx else 'bar'
	color=optx['color'] if 'color' in optx else 'blue'
	width=optx['width'] if 'width' in optx else 0.2
	position=optx['position'] if 'position' in optx else 1.2
	da.plot(ax=ax,kind=kind,position=position,width=width)

	# 2nd plot
	if len(db)>0:
		ax2 = ax.twinx() 
		kind2=optx['kind2'] if 'kind2' in optx else kind
		color2=optx['color2'] if 'color2' in optx else 'red'
		width2=optx['width2'] if 'width2' in optx else width
		position2=optx['position2'] if 'position2' in optx else 0.3
		db.plot(ax=ax2,kind=kind2,color=color2,position=position2,width=width2)

	# plot features
	loc=optx['loc'] if 'loc' in optx else 9
	ncol=optx['ncol'] if 'ncol' in optx else 2
	fancybox=optx['fancybox'] if 'fancybox' in optx else True
	borderaxespad=optx['borderaxespad'] if 'borderaxespad' in optx else 0.
	plt.legend(loc=loc, ncol=ncol, fancybox=fancybox, borderaxespad=borderaxespad)
	rotation=optx['rotation'] if 'rotation' in optx else '30'
	fontsize=optx['fontsize'] if 'fontsize' in optx else 12
	plt.xticks(rotation=rotation,fontsize=fontsize)

	if len(title)>0:
		plt.title(title)
	plt.subplots_adjust(left=0.1,bottom=bottom)
	if chartname is not None and len(chartname)>5:
		plt.savefig(chartpath, format=chartformat)
	elif backend is not None and backend.lower()=='tkagg': 
		plt.show()
	return chartpath,ax,fig

def plot_templates(*args, **opts):
	funcname = getKeyVal(opts,'funcname','plot_ts')
	style = getKeyVal(opts,'style','classic')
	plt.style.use(style)
	if funcname in globals():
		funcArg = globals()[funcname]
	else:
		funcArg = plot_ts
	opts.pop('funcname',None)
	try:
		ret = funcArg(args[0],**opts)
	except Exception as e:
		sys.stderr.write("**ERROR:{} @{}\n".format(str(e),funcname))
		ret= str(e),None,None
	return ret

if __name__ == '__main__':
	from _alan_calc import pullStockHistory as psd
	#df = pd.read_csv("BTCUSD.dat",sep="|")
	#plot_barh2chart(df[['open','close']].tail(), chartname='',backend='tkAgg')
	#plot_ts('^GSPC', chartname='',backend='tkAgg',debugTF=True)
	#plot_ts('CPIAUCNS', chartname='',backend='tkAgg',debugTF=True,src='fred')
	#plot_2yaxis(df[['close']].iloc[:10], backend='tkAgg')
	from _alan_calc import pullStockHistory as psd
	df = psd('^GSPC',gap='1m',ranged='1d')
	plot_ts(df[['close','volume']], chartname='',backend='tkAgg',debugTF=True)
	#plot_intraday_headline(chartname='',backend='tkAgg',debugTF=True)
