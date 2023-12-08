CBECS <- read.csv("C:/Users/FANG/Desktop/506 proj/cbecs2018_final_public.csv", header = TRUE, sep = ",")

## Select predictors of interest
data <- CBECS[ ,c("ELCNS", "SQFT", "WLCNS", "RFCNS", "ELHT1", "ELCOOL", "ELWATR",
                  "ELCOOK", "ELMANU", "CAPGEN","BLDSHP", "YRCONC", "OCCUPYP", 
                  "PBA","CENDIV", "PUBCLIM", "DAYLTP")]


## Overview of data
str(data)
dim(data)

## Change data type of variable
data$WLCNS <- as.factor(data$WLCNS)
data$RFCNS <- as.factor(data$RFCNS)
data$BLDSHP <- as.factor(data$BLDSHP)
data$PBA <- as.factor(data$PBA)
data$CENDIV <- as.factor(data$CENDIV)
data$PUBCLIM <- factor(data$PUBCLIM, ordered = TRUE)
data$ELHT1 <- as.factor(data$ELHT1)
data$ELCOOL <- as.factor(data$ELCOOL)
data$ELWATR <- as.factor(data$ELWATR)
data$ELCOOK <- as.factor(data$ELCOOK)
data$ELMANU <- as.factor(data$ELMANU)
data$CAPGEN <- as.factor(data$CAPGEN)

## Missing value
colSums(is.na(data))

## Missing value imputation
data <- data[,-which(colnames(data)=="OCCUPYP")]
data$DAYLTP[is.na(data$DAYLTP)] <- median(data$DAYLTP, na.rm = TRUE)
data <- data[!is.na(data$ELCNS), ]

## Correlation
cor(data[, sapply(data, is.numeric)], use = "complete.obs")



library(caret)
library(randomForest)
library(xgboost)

## train-test split
#set.seed(123)
index <- createDataPartition(data$ELCNS, p = .8, list = FALSE)
train <- data[index,]
test <- data[-index,]


## Linear Regression Model
lm_model <- lm(ELCNS ~ ., data = train)
lm_pred <- predict(lm_model, newdata = test)


## Random Forest Model
train_control_rf <- trainControl(method = "cv", number = 10)

num_features <- ncol(train) - 1
grid_rf <- expand.grid(mtry = seq(2, num_features, by = 3))

# parameter tuning
rf_tune <- train(
  ELCNS ~ ., 
  data = train, 
  method = "rf",
  trControl = train_control_rf, 
  tuneGrid = grid_rf
)

# Print best parameters
print(rf_tune$bestTune)

# Train the model using the best parameters
rf_model <- randomForest(ELCNS ~ ., data = train, mtry = rf_tune$bestTune$mtry, ntree = 500)

# Make predictions on the test set
rf_pred <- predict(rf_model, newdata = test)


## XGBoost Model
# One-hot encoding
train_matrix <- model.matrix(~ . - 1 - ELCNS, data = train)
test_matrix <- model.matrix(~ . - 1 - ELCNS, data = test)

dtrain <- xgb.DMatrix(data = train_matrix, label = train$ELCNS)
dtest <- xgb.DMatrix(data = test_matrix, label = test$ELCNS)

train_control_xgb <- trainControl(method = "cv", number = 5)

# Define the parameter grid
grid_xgb <- expand.grid(
  nrounds = c(50, 100, 150),
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.05, 0.1),
  gamma = c(0, 0.1, 0.2),
  colsample_bytree = c(0.5, 0.7, 0.9),
  min_child_weight = c(1, 3, 5),
  subsample = c(0.5, 0.7, 0.9)
)

# Parameter tuning 
xgb_tune <- train(
  x = train_matrix, 
  y = train$ELCNS,
  method = "xgbTree",
  trControl = train_control_xgb,
  tuneGrid = grid_xgb,
  metric = "RMSE"
)

# Print best parameters
print(xgb_tune$bestTune)
# Final parameters for the model
final_params <- list(
  max_depth = xgb_tune$bestTune$max_depth,
  eta = xgb_tune$bestTune$eta,
  gamma = xgb_tune$bestTune$gamma,
  colsample_bytree = xgb_tune$bestTune$colsample_bytree,
  min_child_weight = xgb_tune$bestTune$min_child_weight
)

# Train the model with the best parameters
xgb_model <- xgboost(
  data = dtrain, 
  params = final_params, 
  nrounds = xgb_tune$bestTune$nrounds,
  objective = "reg:squarederror",
  verbose = 0
)

# Make predictions on the test set
xgb_pred <- predict(xgb_model, newdata = dtest)



## Model Evaluation
# Residual plot
results <- data.frame(
  Actual = test$ELCNS,
  LM_Residual = lm_pred - test$ELCNS,
  RF_Residual = rf_pred - test$ELCNS,
  XGB_Residual = xgb_pred - test$ELCNS
)

plot1 <- ggplot() +
  geom_point(data = results, aes(x = Actual, y = LM_Residual), color = "blue") +
  ggtitle("Linear Regression Model") +
  xlab("Actual Values") +
  ylab("Residuals") +
  ylim(-5e7, 2.5e7) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  theme_minimal()

plot2 <- ggplot() +
  geom_point(data = results, aes(x = Actual, y = RF_Residual), color = "red") +
  ggtitle("Random Forest Model") +
  xlab("Actual Values") +
  ylab("Residuals") +
  ylim(-5e7, 2.5e7) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  theme_minimal()

plot3 <- ggplot() +
  geom_point(data = results, aes(x = Actual, y = XGB_Residual), color = "green") +
  ggtitle("XGBoost Model") +
  xlab("Actual Values") +
  ylab("Residuals") +
  ylim(-5e7, 2.5e7) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  theme_minimal()

ggpubr::ggarrange(plot1, plot2, plot3, ncol = 3, nrow = 1)

## R squared
lm_r2 <- cor(test$ELCNS, lm_pred)^2
rf_r2 <- cor(test$ELCNS, rf_pred)^2
xgb_r2 <- cor(test$ELCNS, xgb_pred)^2

## RMSE
lm_rmse <- sqrt(mean((lm_pred - test$ELCNS)^2))
rf_rmse <- sqrt(mean((rf_pred - test$ELCNS)^2))
xgb_rmse <- sqrt(mean((xgb_pred - test$ELCNS)^2))

## MAE
lm_mae <- mean(abs(lm_pred - test$ELCNS))
rf_mae <- mean(abs(rf_pred - test$ELCNS))
xgb_mae <- mean(abs(xgb_pred - test$ELCNS))