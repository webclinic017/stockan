#!/usr/bin/env Rscript
#'@program _alan_ohlc_fcs.r
#' Call from _alan_rwrapper.py
#'
#' Time-series 30-day forecast using arima or garch model
#' rAR() is compatible with outlook_forecast() in fa_technical_forecast.R
#' rGARCH() is GARCH model extension verion of rAR() function 
#'
#' Last mod.,
#'  Mon Apr 16 10:33:12 EDT 2018
#===============================================================================#
options(warn=-1)
library(dplyr)
library(tidyr)
library(xts)
library(lubridate)
library(fGarch)
library(forecast)
library(RPostgreSQL)

getData <- function(xqr='', host='localhost',dbname='ara') {
	araDb <- dbConnect(dbDriver("PostgreSQL"), user='sfdbo', host=host, dbname=dbname)
	datax <- dbGetQuery(araDb,xqr)
	return(datax)
}

# Print dataframe to file
prnDf2file<- function(df,collapse="\t",sep="\n",file=stderr()) {
	for(j in 1:NROW(df)) {
		cat(sprintf("%s",paste(df[j,],collapse=collapse)),sep=sep,file=file)
	}
}

# Print vector to file
prnV2file<- function(v,collapse="\t",sep="\n",file=stderr()) {
	cat(sprintf("%s",paste(v,collapse=collapse)),sep=sep,file=file)
}

#'Date format to YYYYMMDD conversion
date2integer <- function(dt)
{
	return( as.integer(as.character(dt,"%Y%m%d")) )
}

#'Create a vector of Dates based on frequency:[freq] as string
#'  and [from] start date in yyyymmdd for [n] period
#'Input: [freq] as one of [D|W|M|Q|Y] character
#'  [from] in yyyymmdd and [n] as integer
#'Output: d as a date vector
freq2dates <- function(freq,from,n)
{
	if(freq=='M') {
		d= ymd(from) %m+% months(seq(1:n))
	} else if(freq=='Q') {
		d= ymd(from) %m+% months(seq(1:n)*3)
	} else if(freq=='Y') {
		d= ymd(from) %m+% months(seq(1:n)*12)
	} else if(freq=='W') {
		d= ymd(from) + (seq(1:n)*7)
	} else if(freq=='D') { # weekday only
		mn=ceiling(n*1.5)
		d= ymd(from) + seq(1:mn)
		d= d[ !(wday(d)%%7 %in% c(1,0)) ]
		d= d[1:n]
	} else {
		d= ymd(from) + seq(1:n)
	}
	return(d)
}

dd2dwm <- function(dfcs,fcsFq=fcsFq,fcsLst=fcsLst,debugTF=debugTF)
{
	cprc=as.numeric(last(dfcs$prc_cur))
	mfcs = dfcs[fcsLst,] %>%	
		mutate(fcsdays=as.integer(fcsLst), freq=fcsFq
		)
	return(mfcs)
}

#'Run rAR/rGARCH for ARIMA/GARCH forecasts for [nfcs] periods 
#'with arima-order:[order] or GARCH formula:fma
#'dependent variable is setup as daily closing prices
#'Input data: [vprc] as closing price vector
#'  and asof date in yyyymmdd as the last observed price date
#'Dependent function: Arima()/garchFit()
#'return: fcsLst list for [dfcs, mfcs]
rForecast <- function(vprc,asof,funcname='rAR',nfcs=30,plevel=0.7,fma="~arma(2,0)+garch(1,1)",order=c(2,0,1),method="CSS",autoArima=F,freq="D",fcsFq=c("D","W","M"),fcsLst=c(1,5,23),logTF=T,difTF=T,dwmTF=T,debugTF=F)
{
	if(debugTF==T){print(tail(vprc,2));}
	if(logTF==T){ rrt=log(vprc);} else { rrt=vprc;}
	if(difTF==T){ rrt=na.omit(diff(rrt)) ;}
	cprc=as.numeric(last(vprc))
	uprc=rev(tail(head(vprc,-1),nfcs))
	prc_chg=cprc/uprc-1
	xchg1d=as.numeric(cprc-head(uprc,1))
	z=qnorm(1-(1-plevel)/2.) # standard [z] score for 2-tail N(0,1) at prob. [plevel]
	xz=ifelse(difTF==T,0,as.numeric(last(rrt))) # USE cprc if 1st-difference is not used
	if(debugTF==T){print(head(rrt,2));}
	if(debugTF==T){print(tail(rrt,3));}
	if(debugTF==T){print("order:");prnV2file(order);}
	afcs=NULL
	if(funcname=='rGARCH') {
		eqtx=as.formula(fma) # use string to set formula
		afit<-garchFit(formula=eqtx,data=rrt,trace=F)
		if(debugTF==T){print(afit);}
		tryCatch({
			afcs=predict(afit,nfcs)
			sd=afcs$meanError #Use meanError OR standardDeviation (smaller)
			fcs1d=afcs$meanForecast
			}, error=function(e){print(e)});
	} else {
		if(autoArima==F) {
			afit=tryCatch(Arima(rrt,order=order,method=method), error=function(e){print(e)});
		} else {
			afit=tryCatch(auto.arima(rrt), error=function(e){print(e)});
		}
		if(debugTF==T){print(arimaorder(afit));}
		if(debugTF==T){print(afit);}
		tryCatch({
			afcs=predict(afit,nfcs)
			sd=afcs$se 
			fcs1d=afcs$pred
			if(debugTF==T){
				print(afcs);
				print(fcs1d); }
			}, error=function(e){print(e)});
		if(is.null(afcs)) { # use forecast() as alternative when predict() is not working for some series (potential package bug)
			afcs = forecast(afit,nfcs,plevel)
			sd=(afcs$upper[,1]-afcs$lower[,1])/z/2 #- redundant verification for rt_sse
			fcs1d=as.numeric(afcs$mean)
		}
	}
	if(debugTF==T){print(afcs);}
	if(difTF==T){
		rt_fcs=cumsum(fcs1d)
		rt_sse=cumsum(sd^2)^.5
	} else {
		rt_fcs=fcs1d
		rt_sse=sd
	}
	pos_pb=1-pnorm((xz-rt_fcs)/rt_sse) # prob. of cumulative positive return
	sigma_plevel=z*rt_sse
	fcsDt=freq2dates(freq,asof,nfcs)
	fcsYmd=date2integer(fcsDt)
	currTime=Sys.time()
	dfcs=data.frame(pbdate=asof,fcsdate=fcsYmd,prc_cur=cprc,xchg1d=xchg1d,prc_x1w=uprc,prc_chg=round(prc_chg,5),freq=freq,
		rt_fcs=rt_fcs,rt_sse=rt_sse,sigma=rt_sse,
		sigma_plevel=sigma_plevel,plevel=plevel,pos_pb=pos_pb,
		lower=rt_fcs-sigma_plevel,upper=rt_fcs+sigma_plevel,last_mod=as.integer(currTime))
	#dfcs=dfcs %>% mutate(xsse=(upper-lower)/z/2) #- redundant verification for rt_sse
	if(debugTF==T){cat(sprintf("xz[%s],nobs[%s] ",xz,length(rrt)),sep="\n",file=stdout())}
	if(debugTF==T){print(head(dfcs,2));}
	if(logTF==T) {
		xprc=ifelse(difTF==T,cprc,1)
		dfcs=dfcs %>%
			mutate( sigma=xprc*(exp(rt_sse)-1),
				prc_fcs=xprc*exp(rt_fcs),
				low_bound=xprc*exp(rt_fcs-sigma_plevel),
				up_bound=xprc*exp(rt_fcs+sigma_plevel)
			)
		if(difTF==F) {
			#mutate( sigma=(exp(2*rt_fcs+rt_sse*rt_sse)*exp(rt_sse*rt_sse-1.))^.5
			dfcs=dfcs %>%
				mutate( sigma=(exp(rt_fcs+rt_sse)-exp(rt_fcs-rt_sse))/2.
				)
		}
		if(debugTF==T){print(head(dfcs,2));}
	} else {
		xprc=ifelse(difTF==T,cprc,0)
		dfcs=dfcs %>%
			mutate( sigma=rt_sse,
				prc_fcs=xprc+(rt_fcs),
				low_bound=xprc+(rt_fcs-sigma_plevel),
				up_bound=xprc+(rt_fcs+sigma_plevel)
			)
		if(debugTF==T){print(head(dfcs,2));}
	}
	dfcs=dfcs %>%
		select(fcsdate,prc_fcs,xchg1d,prc_cur,prc_x1w,prc_chg,sigma_plevel,plevel,up_bound,low_bound,sigma,rrate_fcs=rt_fcs,rrate_sigma=rt_sse,pos_pb,pbdate,last_mod)

	if(debugTF==T){print(head(dfcs,5));}
	if(dwmTF==T) {
	mfcs=dd2dwm(dfcs,fcsFq=fcsFq,fcsLst=fcsLst,debugTF=debugTF);
	}

	fcsLst=list()
	fcsLst[[1]]=dfcs
	if(dwmTF==T) {
	fcsLst[[2]]=mfcs
	}
	return(fcsLst)
}


#'DEPRECATED
#'Run arima forecasts for [nfcs] periods with order:[order] vector
#'  dependent variable is setup as daily return of closing price
#'Input data: [dt] of columns 'close' as closing price 
#'  and date 'pbdate' in yyyymmdd
#'Dependent function: Arima()
#'Output: mfcs
rAR <- function(dt,nobs=90,nfcs=30,p=0.7,order=c(2,0,1),method="CSS",fcsFq=c("D","W","M"),fcsLst=c(1,5,23),debugTF=F)
{
	asof=as.integer(last(dt$pbdate))
	ndts=xts(dt,ymd(dt$pbdate),frequency=364)
	x1wprc=as.numeric(first(tail(ndts$close,6)))
	cprc=as.numeric(last(ndts$close))
	cdate=last(index(ndts))
	#rrt=tail((ndts$close/lag(ndts$close,1)-1),nobs)
	rrt=tail(diff(log(ndts$close)),nobs)
	afit=Arima(rrt,order=order,method=method)
	afcs=predict(afit,nfcs)
	if(debugTF==T){print(head(afcs,7));}
	sd=afcs$se
	fcs1d=afcs$pred

	rfcs=cumsum(fcs1d)
	sse=cumsum(sd^2)^.5
	pos_pb=1-pnorm(-rfcs/sse) # prob. of cumulative positive return
	z=qnorm(1-(1-p)/2.) # standard [z] score for 2-tail N(0,1) at prob. [p]
	pfcs=exp(rfcs)*cprc
	seZ=z*sse
	fcsDt=freq2dates('D',cdate,nfcs)
	fcsYmd=date2integer(fcsDt)
	mfcs=data.frame(asof=asof,fcsdate=fcsYmd,prc_cur=cprc,prc_x1w=x1wprc,rrate_fcs=rfcs,prc_fcs=pfcs,rrate_sigma=sse,sigma_plevel=seZ,plevel=p,pos_pb=pos_pb)
	mfcs = mfcs[fcsLst,] %>%	
		mutate( low_bound=cprc*exp(rrate_fcs-sigma_plevel),
			up_bound=cprc*exp(rrate_fcs+sigma_plevel),
			sigma=cprc*(exp(rrate_sigma)-1),
			fcsdays=fcsLst,
			freq=fcsFq
		)
	return(mfcs)
}

#'DEPRECATED
#'Run GARCH forecasts for [nfcs] periods with formula:[fma]
#'  dependent variable is setup as daily return of closing price
#'Input data: [dt] of columns 'close' as closing price 
#'  and date 'pbdate' in yyyymmdd
#'Dependent function: garchFit()
#'  with relevant variables: meanForecast,meanError,standardDeviation
#'Output: mfcs
rGARCH <- function(dt,nobs=90,nfcs=30,p=.70,fma="~arma(2,0)+garch(1,1)",fcsFq=c("D","W","M"),fcsLst=c(1,5,23),debugTF=F)
{
	if(debugTF==T){print(names(dt));print(p)}
	asof=as.integer(last(dt$pbdate))
	ndts=xts(dt,ymd(dt$pbdate),frequency=364)
	x1wprc=as.numeric(first(tail(ndts$close,6)))
	cprc=as.numeric(last(ndts$close))
	cdate=last(index(ndts))
	#rrt=tail((ndts$close/lag(ndts$close,1)-1),nobs)
	rrt=tail(diff(log(ndts$close)),nobs)
	eqtx=as.formula(fma) # use string to set formula
	afit<-garchFit(formula=eqtx,data=rrt,trace=F)
	#afit<-garchFit(formula=~arma(2,0)+garch(1,1),data=rrt,trace=F)
	afcs=predict(afit,nfcs)
	if(debugTF==T){print(head(afcs,7));}
	sd=afcs$standardDeviation #Use meanError OR standardDeviation (smaller)
	fcs1d=afcs$meanForecast

	rfcs=cumsum(fcs1d)
	sse=cumsum(sd^2)^.5
	pos_pb=1-pnorm(-rfcs/sse) # prob. of cumulative positive return
	z=qnorm(1-(1-p)/2.) # standard [z] score for 2-tail N(0,1) at prob. [p]
	pfcs=exp(rfcs)*cprc
	seZ=z*sse
	fcsDt=freq2dates('D',cdate,nfcs)
	fcsYmd=date2integer(fcsDt)
	mfcs=data.frame(asof=asof,fcsdate=fcsYmd,prc_cur=cprc,prc_x1w=x1wprc,rrate_fcs=rfcs,prc_fcs=pfcs,rrate_sigma=sse,sigma_plevel=seZ,plevel=p,pos_pb=pos_pb)
	mfcs = mfcs[fcsLst,] %>%	
		mutate( low_bound=cprc*exp(rrate_fcs-sigma_plevel),
			up_bound=cprc*exp(rrate_fcs+sigma_plevel),
			sigma=cprc*(exp(rrate_sigma)-1),
			fcsdays=fcsLst,
			freq=fcsFq
		)
	return(mfcs)
}
