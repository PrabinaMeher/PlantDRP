

library(xgboost)
library(magrittr)
library(Matrix)
library(Biostrings)
library(PSSMCOOL)
library(R2HTML)


setwd("C:/xampp/htdocs/plantdrp/server")#change
#####################################################
if (file.exists("drp.html")) file.remove("drp.html")
if (file.exists("comp_pssm.txt")) file.remove("comp_pssm.txt")
if (file.exists("dp_pssm.txt")) file.remove("dp_pssm.txt")
if (file.exists("ksb_pssm.txt")) file.remove("ksb_pssm.txt")


####################################################################
###########Sequence filtering##########
x <- readAAStringSet("example.fasta")
nx <- names(x)
names(x)<- nx
writeXStringSet(x, "test.fasta")

xx <- toupper(as.character(as.character(x)))
#Checking of standard residues#
std <- c("A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y")
zx <- xx
z <-  sapply(zx, function(s) strsplit(s, split=""))
zz <- lapply(z, table)
zz1 <- unique(as.character(unlist(lapply(zz, names))))


if(length(union(zz1, std))>20){
  pp <- data.frame("Contain non-standard residues,so kindly submit sequences having standard residues only")
  names(pp)<- "error message"
  HTML(pp,"drp.html")
  
}else{
  
COMP_PSSM <- matrix(0, nrow=length(x), ncol=400)
DP_PSSM <- matrix(0, nrow=length(x), ncol=240)
KSB_PSSM <- matrix(0, nrow=length(x), ncol=400)
  
  for(i in 1:length(x)){
    
    writeXStringSet(x[i],"query.fasta")
    shell("C:/xampp/htdocs/plantdrp/server/bin/psiblast -query C:/xampp/htdocs/plantdrp/server/query.fasta -db C:/xampp/htdocs/plantdrp/server/train -num_iterations 3 -evalue 0.001 -out_ascii_pssm C:/xampp/htdocs/plantdrp/server/protein.pssm")
    
   	
	  COMP_PSSM[i,]<-as.matrix(suppressWarnings(as.numeric(pssm_composition("protein.pssm"))), ncol=400, nrow=1)
    DP_PSSM[i,]<-as.matrix(suppressWarnings(as.numeric(DP_PSSM("protein.pssm"))), ncol=240, nrow=1)
    KSB_PSSM[i,]<-as.matrix(suppressWarnings(as.numeric(k_separated_bigrams_pssm("protein.pssm", k=1))), ncol=400, nrow=1)

    
  }
  
write.table(COMP_PSSM,"comp_pssm.txt", row.names=FALSE, col.names=FALSE, sep="\t")
write.table(DP_PSSM,"dp_pssm.txt", row.names=FALSE, col.names=FALSE, sep="\t")
write.table(KSB_PSSM,"ksb_pssm.txt", row.names=FALSE, col.names=FALSE, sep="\t")

pp1 <- read.table("comp_pssm.txt")
pp2 <- read.table("dp_pssm.txt")
pp3 <- read.table("ksb_pssm.txt")


feat <- read.table("drg_xgboost_feature.txt")[,1]
ft <- feat[1:105]


tst<-cbind(pp1,pp2,pp3)
colnames(tst) <- as.numeric(1:ncol(tst))
tst <- tst[,ft]

c1 <- rep(1, nrow(tst)) 
dat_ts <- cbind(c1, tst)
colnames(dat_ts)<- paste("Species_",seq(ncol(dat_ts)), sep="")
dat_ts[,1] <- as.integer(dat_ts[,1])


load_booster <- xgb.load("xgb.model")
p_pos <- predict(load_booster, newdata=as.matrix(dat_ts[,-1]))


kk <- which(p_pos>=0.5)
lab <- vector(mode="character", length=length(x))
lab[kk]<- "DRP"
lab[-kk] <- "non-DRP"

p_pos[which (p_pos<0.5)] <- 1- p_pos[which (p_pos<0.5)]
      
res <- data.frame(nx, lab, p_pos)
colnames(res) <- c("Sequence ID", "Predicted as", "With probability")
  HTML(res,file="drp.html", align="center",Border = 1, innerBorder = 1)
}
