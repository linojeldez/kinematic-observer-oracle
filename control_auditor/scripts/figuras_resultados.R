args <- commandArgs(trailingOnly=FALSE)
script <- sub("^--file=", "", args[grep("^--file=",args)])
script <- gsub("~\\+~", " ", script)
root <- dirname(dirname(normalizePath(script)))
dir.create(file.path(root,"figuras"),showWarnings=FALSE)
pal <- c("#2166AC","#D6604D","#1B7837")

if (file.exists(file.path(root,"resultados","contrastes_principales.csv"))) {
c <- read.csv(file.path(root,"resultados","contrastes_principales.csv"),stringsAsFactors=FALSE)
names_es <- c(fixed="Misma ganancia e intercepto",fpr="FPR equiparada",mean="Atencion media equiparada",joint="FPR y atencion equiparadas")
labs <- paste(ifelse(c$protocol=="forzado","Forzado","Maniobra"),names_es[c$scheme],sep=" / ")
png(file.path(root,"figuras","contrastes_principales.png"),width=2200,height=1250,res=190)
par(mar=c(5,21,4,2),mgp=c(3,1,0))
y <- rev(seq_len(nrow(c)))
lo <- c$simultaneous_CI95_lo;hi <- c$simultaneous_CI95_hi
lim <- range(c(0,lo,hi));span <- diff(lim)
plot(c$difference_R,y,xlim=lim+c(-.06,.1)*span,ylim=c(.5,nrow(c)+.5),pch=16,col=pal[1],cex=1.3,
     yaxt="n",ylab="",xlab="Diferencia de responsividad: memoria 1 s menos instantaneo",main="Memoria con el mismo estadistico Student-t")
abline(v=0,lty=2,col="grey45")
abline(h=4.5,col="grey85")
segments(lo,y,hi,y,lwd=2.5,col=pal[1]);segments(lo,y-.08,lo,y+.08,col=pal[1]);segments(hi,y-.08,hi,y+.08,col=pal[1])
axis(2,at=y,labels=labs,las=1,cex.axis=.92,tick=FALSE)
mtext("IC simultaneos 95% de ocho contrastes; ocho replicas pareadas, 18 angulos por replica",side=3,line=.35,cex=.8)
dev.off()
}

if (file.exists(file.path(root,"resultados","resumen_condiciones.csv"))) {
s <- read.csv(file.path(root,"resultados","resumen_condiciones.csv"),stringsAsFactors=FALSE)
s <- subset(s,scheme=="fixed" & stat=="full" & tauW==4)
png(file.path(root,"figuras","factorial_memoria_ganancia.png"),width=2200,height=1100,res=180)
par(mfrow=c(1,2),mar=c(5,5,4,1),oma=c(1,0,0,0))
ys <- range(100*c(s$recovery_lo,s$recovery_hi),na.rm=TRUE)
if (!all(is.finite(ys))) ys <- c(0,100)
for(protocol in c("forzado","maniobra")) {
 d <- s[s$protocol==protocol,,drop=FALSE]
 plot(0,0,type="n",xlim=c(-.12,4.15),ylim=ys,xaxt="n",xlab="Memoria adicional (s)",ylab="Ventaja del oraculo recuperada (%)",
      main=ifelse(protocol=="forzado","Lider forzado","Una maniobra"))
 axis(1,at=c(0,.25,1,4),labels=FALSE)
 mtext(c("0","0.25","1","4"),side=1,at=c(0,.25,1,4),line=.85,cex=.85)
 abline(h=c(0,100),col="grey85",lty=c(1,2))
 for(i in seq_along(sort(unique(d$gain)))) {
   g <- sort(unique(d$gain))[i];dd <- d[d$gain==g,];dd <- dd[order(dd$tau),]
   lines(dd$tau,100*dd$recovery,col=pal[i],lwd=2,type="b",pch=c(16,17,15)[i])
   segments(dd$tau,100*dd$recovery_lo,dd$tau,100*dd$recovery_hi,col=pal[i])
 }
 legend("topright",legend=paste("alpha =",format(sort(unique(d$gain)),trim=TRUE)),col=pal,lty=1,pch=c(16,17,15),bty="n",cex=.8)
}
mtext("Factorial complementario: 4 replicas; celdas centrales a alpha 12.048 y memoria 0/1 s: 8 replicas. IC individuales.",outer=TRUE,side=1,line=0,cex=.7)
dev.off()
}

n <- read.csv(file.path(root,"calibracion","resumen_validacion.csv"),stringsAsFactors=FALSE)
png(file.path(root,"figuras","validacion_nula.png"),width=2100,height=1000,res=180)
par(mfrow=c(1,2),mar=c(5,5,4,1),oma=c(2,0,0,0))
schemes <- c("fpr","mean","joint")
for(metric in c("fpr","attention")) {
 low <- n[[paste0(metric,"_low")]];high <- n[[paste0(metric,"_high")]]
 target <- ifelse(metric=="fpr",.05,.173)
 plot(0,0,type="n",xlim=c(.6,3.4),ylim=range(c(0,high))*1.1,xaxt="n",xlab="Regla de calibracion",ylab=ifelse(metric=="fpr","Fraccion de pesos mayores de 0.5","Peso medio"),
      main=ifelse(metric=="fpr","Falsos positivos basales","Atencion basal"))
 axis(1,at=1:3,labels=c("FPR","Media","Ambas"))
 abline(h=target,lty=2,col="grey40")
 for(i in seq_along(c(0,1))) {
   d <- n[n$tau==c(0,1)[i],];pos <- match(d$scheme,schemes)+c(-.08,.08)[i]
   points(pos,d[[metric]],pch=c(16,17)[i],col=pal[i],cex=1.2)
   segments(pos,d[[paste0(metric,"_low")]],pos,d[[paste0(metric,"_high")]],col=pal[i],lwd=2)
 }
 legend("topleft",legend=c("Instantaneo","Memoria 1 s"),col=pal[1:2],pch=c(16,17),bty="n",cex=.8)
}
mtext("Ocho semillas nulas retenidas, sin reajuste. Intervalos por semilla. Lineas: objetivos operacionales.",outer=TRUE,side=1,line=.3,cex=.8)
dev.off()
