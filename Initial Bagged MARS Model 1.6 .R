# ===============================
# FULL BAGGED MARS FRAUD MODEL
# ===============================

# 1. LOAD LIBRARIES
library(caret)
library(earth)
library(dplyr)
library(pROC)

# 2. LOAD DATA
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
# 4. TARGET ENCODING (High-cardinality columns)
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

# Numeric columns (exclude encoded + target)
raw_numeric_cols <- names(df_clean)[sapply(df_clean, is.numeric) & 
                                      !grepl("_Encoded$", names(df_clean)) & 
                                      names(df_clean) != "Is_Fraud"]

# Center & scale
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
# 7. TRAIN/TEST SPLIT
# ===============================
set.seed(123)
trainIndex <- createDataPartition(final_y, p = 0.8, list = FALSE)

trainX <- final_x[trainIndex, ]
trainY <- final_y[trainIndex]

testX <- final_x[-trainIndex, ]
testY <- final_y[-trainIndex]

# ===============================
# 8. BAGGED (BOOTSTRAP) MARS
# ===============================
set.seed(123)
B <- 40  # Number of bootstrap models
pred_matrix <- matrix(NA, nrow = nrow(testX), ncol = B)

cat("Training Bagged MARS...\n")
for (b in 1:B) {
  
  cat("Iteration:", b, "\n")
  
  # Bootstrap sample
  idx <- sample(1:nrow(trainX), replace = TRUE)
  bootX <- trainX[idx, ]
  bootY <- trainY[idx]
  
  # Balance dataset
  fraud_idx <- which(bootY == 1)
  non_fraud_idx <- sample(which(bootY == 0), min(20000, sum(bootY == 0)))
  sub_idx <- c(fraud_idx, non_fraud_idx)
  
  bootX_sub <- bootX[sub_idx, ]
  bootY_sub <- bootY[sub_idx]
  
  weights_sub <- ifelse(bootY_sub == 1, 5, 1)
  
  # Train MARS
  model <- earth(
    x = bootX_sub,
    y = bootY_sub,
    degree = 2,
    nk = 40,
    nprune = 20,
    weights = weights_sub,
    glm = list(family = binomial)
  )
  
  # Predict
  pred_matrix[, b] <- predict(model, newdata = testX, type = "response")
}

# Average predictions
probs <- rowMeans(pred_matrix)

# ===============================
# 9. FRAUD-FOCUSED EVALUATION
# ===============================
hist(probs, breaks=50, col="skyblue", main="Fraud Probability Distribution")

# Threshold (can tune later)
threshold <- quantile(probs, 0.95)

predicted_class <- factor(ifelse(probs > threshold, 1, 0), levels = c("0", "1"))
testY <- factor(testY, levels = c("0", "1"))  # ensure levels correct

conf_matrix <- confusionMatrix(predicted_class, testY, positive = "1")
print(conf_matrix)

# Fraud-focused metrics
precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Sensitivity"]
f1 <- 2 * (precision * recall) / (precision + recall)

cat("\n--- FRAUD METRICS ---\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1, "\n")

# ===============================
# 10. ROC CURVE
# ===============================
roc_obj <- roc(testY, as.numeric(probs))
plot(roc_obj, col="blue", lwd=2,
     main=paste("Bagged MARS ROC - AUC:", round(auc(roc_obj), 3)))
abline(a=0, b=1, lty=2, col="red")
cat("AUC:", auc(roc_obj), "\n")
