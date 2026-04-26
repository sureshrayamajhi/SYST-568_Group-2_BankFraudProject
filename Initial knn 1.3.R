# ===============================

# FULL OPTIMIZED KNN FRAUD MODEL

# ===============================

# ===============================

# 1. LOAD LIBRARIES

# ===============================

library(caret)
library(dplyr)
library(pROC)

# ===============================

# 2. LOAD DATA

# ===============================

df <- read.csv("new_bank_fraud_detection.csv", stringsAsFactors = FALSE)

# ===============================

# 3. FEATURE ENGINEERING

# ===============================

# --- Time features

df$Transaction_Time <- as.POSIXct(df$Transaction_Time, format = "%H:%M:%S")
df$Transaction_Hour <- as.numeric(format(df$Transaction_Time, "%H"))
df$Transaction_Time <- NULL

# --- Date features

df$Transaction_Date <- as.Date(as.character(df$Transaction_Date), format="%d-%m-%Y")
df$Transaction_Weekday <- weekdays(df$Transaction_Date)

# --- Weekend indicator

df$Is_Weekend <- ifelse(df$Transaction_Weekday %in% c("Saturday", "Sunday"), 1, 0)
df$Transaction_Weekday <- as.factor(df$Transaction_Weekday)
df$Transaction_Date <- NULL

# --- Remove identifiers

df <- df[, !(names(df) %in% c("Transaction_ID", "Merchant_ID", "X", "Transaction_Currency"))]

# --- Binary encoding

df$Gender <- ifelse(df$Gender == "Female", 1, 0)

# --- Amount / Balance ratio

df$Amt_Balance_Ratio <- df$Transaction_Amount / (df$Account_Balance + 1)
df$Amt_Balance_Ratio[is.infinite(df$Amt_Balance_Ratio)] <- 0
df$Amt_Balance_Ratio[is.na(df$Amt_Balance_Ratio)] <- 0

# ===============================

# 4. TARGET ENCODING

# ===============================

high_card_vars <- c("State", "Bank_Branch", "Transaction_Location", "Transaction_Description")

for (col in high_card_vars) {
  target_means <- df %>%
    group_by(across(all_of(col))) %>%
    summarise(mean_target = mean(as.numeric(as.character(Is_Fraud)), na.rm = TRUE), .groups = 'drop')
  
  df[[paste0(col, "_Encoded")]] <- target_means$mean_target[match(df[[col]], target_means[[col]])]
}

df_clean <- df[, !(names(df) %in% high_card_vars)]

# ===============================

# 5. PREPROCESSING

# ===============================

final_y <- as.factor(df_clean$Is_Fraud)

raw_numeric_cols <- names(df_clean)[sapply(df_clean, is.numeric) &
                                      !grepl("_Encoded$", names(df_clean)) &
                                      names(df_clean) != "Is_Fraud"]

# Scale numeric features

pp <- preProcess(df_clean[, raw_numeric_cols], method = c("center", "scale"))
df_clean[, raw_numeric_cols] <- predict(pp, df_clean[, raw_numeric_cols])

df_clean[is.na(df_clean)] <- 0

# ===============================

# 6. DUMMY ENCODING

# ===============================

predictors_only <- df_clean[, names(df_clean) != "Is_Fraud"]
dummies_model <- dummyVars(~ ., data = predictors_only)
final_x <- as.data.frame(predict(dummies_model, newdata = predictors_only))

# ===============================

# 7. TRAIN / TEST SPLIT

# ===============================

set.seed(123)
trainIndex <- createDataPartition(final_y, p = 0.8, list = FALSE)

trainX <- final_x[trainIndex, ]
trainY <- final_y[trainIndex]

testX <- final_x[-trainIndex, ]
testY <- final_y[-trainIndex]

# Fix class labels (IMPORTANT)

trainY <- factor(ifelse(trainY == 1, "Fraud", "NonFraud"),
                 levels = c("NonFraud", "Fraud"))

testY  <- factor(ifelse(testY == 1, "Fraud", "NonFraud"),
                 levels = c("NonFraud", "Fraud"))

# ===============================

# 8. PCA (MAJOR SPEED BOOST)

# ===============================

pp_pca <- preProcess(trainX, method = "pca", thresh = 0.95)

trainX <- predict(pp_pca, trainX)
testX  <- predict(pp_pca, testX)

cat("Number of features after PCA:", ncol(trainX), "\n")

# ===============================

# 9. KNN MODEL (OPTIMIZED)

# ===============================

set.seed(123)

ctrl <- trainControl(
  method = "cv",
  number = 3,              # reduced from 5
  sampling = "smote",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

knn_model <- train(
  x = trainX,
  y = trainY,
  method = "knn",
  tuneLength = 5,          # reduced from 10
  trControl = ctrl,
  metric = "ROC"
)

cat("Best k:\n")
print(knn_model$bestTune)

# ===============================

# 10. PREDICTIONS

# ===============================

knn_probs <- predict(knn_model, newdata = testX, type = "prob")[, "Fraud"]

# Fraud-focused threshold

threshold <- quantile(knn_probs, 0.95)

knn_pred <- factor(ifelse(knn_probs > threshold, "Fraud", "NonFraud"),
                   levels = c("NonFraud", "Fraud"))

# ===============================

# 11. EVALUATION

# ===============================

conf_matrix <- confusionMatrix(knn_pred, testY, positive = "Fraud")
print(conf_matrix)

precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Sensitivity"]
f1 <- 2 * (precision * recall) / (precision + recall)

cat("\n--- KNN FRAUD METRICS ---\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1, "\n")

# ===============================

# 12. ROC CURVE

# ===============================

roc_obj <- roc(testY, knn_probs)

plot(roc_obj, col="green", lwd=2,
     main=paste("KNN ROC - AUC:", round(auc(roc_obj), 3)))
abline(a=0, b=1, lty=2, col="red")

cat("AUC:", auc(roc_obj), "\n")
