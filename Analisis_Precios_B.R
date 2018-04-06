

# crim: per capita crime rate by town.
# zn: proportion of residential land zoned for lots over 25,000 sq.ft.
# indus: proportion of non-retail business acres per town.
# chas: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# nox: nitrogen oxides concentration (parts per 10 million).
# rm: average number of rooms per dwelling.
# age: proportion of owner-occupied units built prior to 1940.
# dis: weighted mean of distances to five Boston employment centres.
# rad: index of accessibility to radial highways.
# tax: full-value property-tax rate per $10,000.
# ptratio: pupil-teacher ratio by town.
# black: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
# lstat: lower status of the population (percent).
# medv: median value of owner-occupied homes in $1000s.

#Borrar todas las variables cargadas.
rm(list=ls()) 

library(C50)
library(caTools) 
library(ggplot2)
library(lattice) 
library(caret)
library(corrplot) 
library(car)
library(class)
library(dplyr)
library(gridExtra)
library(gmodels)
library(ISLR)
library(plyr)
library(psych)
library(RWeka)
library(tree)
library(MASS)

#Cargar el dataset
data(Boston)      #Service concerning housing in the area of Boston Mass.
View(Boston)
sapply(Boston, class)
str(Boston)
summary(Boston)
length(Boston)
attach(Boston)

# Analisis de correlacion.
cr<-cor(Boston)  
cr
pairs.panels(Boston, col="red",cex.labels=1.3,cex.cor = 2)

#Scatter Plot Matrices
splom(~Boston[c(1:7)],groups=NULL,data=Boston,axis.line.tck=0,axis.text.alpha=0,col="red")
splom(~Boston[c(8:14)],groups=NULL,data=Boston,axis.line.tck=0,axis.text.alpha=0,col="blue")

corrplot(cr,type="lower")
corrplot(cr,method="number",title = "Correlacion variables precio casas.")

# Variables con mayor correlación. 
plot(crim,medv,cex=0.5,xlab="Tasa de criminalidad.",ylab="Precio en miles de USD.")
abline(lm(medv~crim),col="red")

plot(rm,medv,cex=1,xlab="Habitaciones por vivienda.",ylab="Precio en miles de USD.")
abline(lm(medv~rm),col="green")

plot(medv,lstat,cex=1,xlab="Precio en miles de USD.",ylab="Estrato social.")
abline(lm(lstat~medv),col="blue")

plot(indus,nox,cex=1,xlab="Industria.",ylab="Concentración N2.")
abline(lm(nox~indus),col="gold")

############# PARTE 1: Prediccion Precios  c5.0, Quinlan #############
summary(medv)

Boston$precio<-"bajo"
Boston$precio<-ifelse(Boston$medv>mean(Boston$medv),"alto",Boston$precio)
Boston$precio<- factor(Boston$precio)

######################################### Conjunto de entrenamiento y prueba.

set.seed(15)
train.size <- 0.77 
Boston.index <- sample.int(length(Boston$medv), round(length(Boston$medv) * train.size))
Boston.train<- Boston[Boston.index,]
Boston.test <- Boston[-Boston.index,]

### Verificar la proporcion de valores.
prop.table(table(Boston.train$precio))*100
prop.table(table(Boston.test$precio))*100

#########################################

#Entrenamiento del modelo.
#Excluir la variable objetivo, precio y medv.
colnames(Boston.train)
BostonP.model <- C5.0(Boston.train[,c(-14,-15)], Boston.train$precio)
BostonP.model
summary(BostonP.model)

### Evaluacion del desempeño del modelo. 
#Aplicacion del modelo en el conjunto de prueba.
Boston.pred <- predict(BostonP.model, Boston.test) 

#Resultados obtenidos
CrossTable(Boston.test$precio,Boston.pred,prop.chisq = FALSE, prop.c = FALSE,
           prop.r = FALSE, dnn = c('Valor real', 'Valor predecido'))

# Mejorar el desempeño del modelo.
# Adicional parametros de prueba que indiquen el número de arboles de desiciones a usar.
### Penalizacion por diferentes tipos de error.
### Matriz de costo
matrix_dimensions <- list(c("bajo", "alto"), c("bajo", "alto"))
names(matrix_dimensions) <- c("Predecido", "Actual")
matrix_dimensions

### Asignacion de los valores de las penalizaciones.

# Falsos positivos, peso de 2.
# Falso Negativo, peso de 4. 

error_cost <- matrix(c(0, 6, 1, 0), nrow = 2, dimnames = matrix_dimensions)
error_cost

### Se vuelve a predecir el valor.
BostonP.model1 <- C5.0(Boston.train[,c(-14,-15)], Boston.train$precio,costs = error_cost,trials = 10)
Boston.pred1 <- predict(BostonP.model1, Boston.test)

#Resultados obtenidos
CrossTable(Boston.test$precio,Boston.pred1,prop.chisq = FALSE, prop.c = FALSE,
           prop.r = FALSE, dnn = c('Valor real', 'Valor predecido'))

############# Reduccion del numero de variables a tomar en cuenta. 

BostonR<-Boston[,c("lstat","indus","rm","age","crim","medv","precio")]

crR<-cor(BostonR[,c("lstat","indus","rm","age","crim","medv")])  # Calcula la correlación del dataset reducido.
crR
corrplot(crR,method="number")
pairs.panels(BostonR, col="blue",cex.labels=1.3,cex.cor = 1.7)

#### El proceso se repite nuevamente para el conjunto reducido. 

BostonR.index <- sample.int(length(BostonR$medv), round(length(BostonR$medv) * train.size))
BostonR.train <- BostonR[ Boston.index,]
BostonR.test  <- BostonR[-Boston.index,]

### Verificar la proporcion de valores.
prop.table(table(BostonR.train$precio))*100
prop.table(table(BostonR.test$precio))*100

### Modelo prediccion Reducido.
BostonR.model <- C5.0(BostonR.train[,c(-14,-15)], BostonR.train$precio,costs = error_cost,trials = 10)
summary(BostonR.model)
BostonR.model
BostonR.pred <- predict(BostonR.model, BostonR.test)

#Resultados obtenidos
CrossTable(BostonR.test$precio,BostonR.pred,prop.chisq = FALSE, prop.c = FALSE,
           prop.r = FALSE, dnn = c('Valor real', 'Valor predecido'))

# Los resultados mejoran al no considerar 
# aquellas variables con baja correlacion con el factor a predecir

######################## FIN PARTE 1 ########################

############### Parte 2 k-Nearest Neighbour Classification ############### 

BostonN<-Boston # Usaremos este nuevo conjunto para el analisis. 

table(BostonN$precio)
# Obtener el porcentaje de casas con un precio alto y bajo.
round(prop.table(table(BostonN$precio))*100,digits = 1)

# Función de normalización.
normalizar<-function(x){
  return((x-min(x))/(max(x)-min(x)))
}

# Normalizacion de los valores del dataframe
BostonN<-as.data.frame(lapply(BostonN[1:14],normalizar))
summary(BostonN)
View(BostonN)

##################### Conjunto de entrenamiento y prueba. ####################

BostonN.index <- sample.int(length(BostonN$medv), round(length(BostonN$medv) * train.size))
BostonN.train <- BostonN[BostonN.index,]
BostonN.test  <- BostonN[-BostonN.index,]

### Verificar la proporcion de valores.
round(prop.table(table(BostonN.train$medv>mean(BostonN$medv)))*100,digits = 1)
round(prop.table(table(BostonN.test$medv>mean(BostonN$medv)))*100,digits = 1)

#Obtencion de los valores originales para validar el modelo.
train.labels<-Boston[BostonN.index,15]
test.labels <-Boston[-BostonN.index,15]

#Modelo de entrenamiento

BostonN.model<-knn(train = BostonN.train,test = BostonN.test,cl = train.labels,k=21)
BostonN.model

# Evaluacion del modelo mediante una tabla crusada.

CrossTable(x=test.labels,y=BostonN.model,prop.chisq = FALSE)

################### FIN PARTE 2 ###################

############### Parte 3 Weka_classifier_rules {RWeka} ############### 

BostonW<-Boston

BostonW<-BostonW[,c(-14)]

View(BostonW)
table(BostonW$precio)
BostonW$precio<-as.factor(BostonW$precio)
sapply(BostonW, class)

# Emplear el modelo 1R para predecir el comportamiento
BostonW_1R<-OneR(precio~.,data=BostonW)
BostonW_1R
summary(BostonW_1R)

# Mejora el desempeño del modelo # JRip() modelo basado en Java implementado en Ripper
BostonW_JRip<-JRip(precio~.,data=BostonW)
BostonW_JRip
summary(BostonW_JRip)


################### FIN PARTE 3 ###################

############### Parte 4 Arbol de clasificación {tree} ############### 

data(Boston) 

#Genera un vector aleatorio de 253 elementos con valores entre el 1:506.
train=sample(1:nrow(Boston),nrow(Boston)/2)   
test=-train

training_data=Boston[train,] # Conjunto de entrenamiento
testing_data=Boston[test,]   # Conjunto de prueba
testing_medv=medv[test]      # Median value of owner-occupied homes in \$1000s.


#Ajuste del arbol en funcion de los datos de entrenamiento. 
tree_model=tree(medv~., training_data)
tree_model
plot(tree_model)
text(tree_model,pretty=0)

#Validar el modelo con la informacion de prueba. 
tree_pred=predict(tree_model, testing_data)
summary(tree_pred)
mean((tree_pred-testing_medv)^2)
#table(tree_pred, testing_medv)

# Cross validation for pruning the tree

cv_tree=cv.tree(tree_model)
plot(cv_tree$size, cv_tree$dev,type="b",xlab="Tree Size", ylab="MSE")

which.min(cv_tree$dev)
cv_tree$size[1]


#prune the tree to size 4

pruned_model=prune.tree(tree_model, best=4)
plot(pruned_model)
text(pruned_model,pretty=0)

#Check the accuracy of the model using testing data

tree_pred1=predict(pruned_model, testing_data)
mean((tree_pred1-testing_medv)^2)


