# ==========================================
# 1. SETUP & UPDATED PREDICTORS
# ==========================================
suppressPackageStartupMessages({
  library(AppliedPredictiveModeling)
  library(caret)
  library(e1071)
  library(corrplot)
  library(pROC)
  library(earth)      # MARS
  library(dplyr)
})

set.seed(123)
df_raw <- read.csv("new_bank_fraud_detection.csv", stringsAsFactors = FALSE)

# --- FEATURE SYNTHESIS (Updating Predictors) ---
df <- df_raw %>%
  mutate(
    # Time and Interaction Signals
    Hour = as.numeric(format(as.POSIXct(Transaction_Time, format="%H:%M:%S"), "%H")),
    Is_Night = ifelse(Hour >= 0 & Hour <= 5, 1, 0),
    Amt_Bal_Ratio = Transaction_Amount / (Account_Balance + 1),
    Log_Amt = log1p(Transaction_Amount),
    Amt_Night_Interaction = Log_Amt * Is_Night,
    # Binary Target
    Is_Fraud_Num = Is_Fraud,
    Is_Fraud = factor(Is_Fraud, levels = c(0, 1), labels = c("Normal", "Fraud"))
  )

# --- TARGET ENCODING (High-Cardinality Predictors) ---
target_cols <- c("State", "Bank_Branch", "Transaction_Location")
for (col in target_cols) {
  stats <- df %>% group_by(.data[[col]]) %>% 
    summarise(Risk = mean(Is_Fraud_Num, na.rm=TRUE), .groups='drop')
  df[[paste0(col, "_Risk")]] <- stats$Risk[match(df[[col]], stats[[col]])]
}

# ==========================================
# 2. CLEANING & FILTERING (Ref. Steps 3-7)
# ==========================================
# 3. Remove Identifier Columns
df <- df[, !(names(df) %in% c("Transaction_ID", "Merchant_ID", "X"))]

# 4. Convert Binary Gender (Numeric Dummy)
df$Gender <- ifelse(df$Gender == "Female", 1, 0)

# 6. Separate Predictors & Numeric Filtering
fraudX_num <- df[, sapply(df, is.numeric)]
fraudX_num <- fraudX_num[, !(names(fraudX_num) %in% "Is_Fraud_Num")]

# 7. Remove Near-Zero Variance
nzv <- nearZeroVar(fraudX_num)
if(length(nzv) > 0) fraudX_num <- fraudX_num[, -nzv]

# ==========================================
# 3. TRANSFORMATION & VISUALS (Ref. Steps 8-10)
# ==========================================
# 8. Check Skewness
cat("\n--- SKEWNESS REPORT ---\n")
print(apply(fraudX_num, 2, skewness))

# 9. Yeo-Johnson Transformation (Ref Logic)
preProc <- preProcess(fraudX_num, method = c("YeoJohnson", "center", "scale"))
fraudX_trans <- predict(preProc, fraudX_num)

# Boxplot: Showing the effect of the updated predictors
par(mfrow=c(1,2))
boxplot(df$Amt_Bal_Ratio ~ df$Is_Fraud, main="Ratio (Raw)", col="gold")
boxplot(fraudX_trans$Amt_Bal_Ratio ~ df$Is_Fraud, main="Ratio (Transformed)", col="cyan")

# 10. Correlation Analysis (Ref Logic)
corrMatrix <- cor(fraudX_trans)
par(mfrow=c(1,1))
corrplot(corrMatrix, order = "hclust", tl.cex = 0.6, main = "Updated Predictor Correlation")

# Filter High Correlation (> 0.75)
highCorr <- findCorrelation(corrMatrix, cutoff = 0.75)
if(length(highCorr) > 0) {
  fraudX_filtered <- fraudX_trans[, -highCorr]
} else {
  fraudX_filtered <- fraudX_trans
}

# ==========================================
# 4. PCA & DATA RECOMBINATION (Ref. Steps 11-13)
# ==========================================
# 11. PCA Deep Dive
fraudPCA <- prcomp(fraudX_filtered, center = FALSE, scale. = FALSE)

cat("\n--- PCA LOADINGS: WHAT'S INSIDE PC1? ---\n")
print(sort(abs(fraudPCA$rotation[,1]), decreasing = TRUE)[1:5])

# 12. Recombine Clean Dataset
finalData <- data.frame(fraudX_filtered, Is_Fraud = df$Is_Fraud, Is_Fraud_Num = df$Is_Fraud_Num)

# 13. Structure Check
cat("\n--- FINAL STRUCTURE CHECK ---\n")
str(finalData)

# ==========================================
# 5. MARS MODEL (Weighted earth process)
# ==========================================
set.seed(123)
trainIdx <- createDataPartition(finalData$Is_Fraud, p = 0.8, list = FALSE)
train_set <- finalData[trainIdx, ]
test_set  <- finalData[-trainIdx, ]

# Weighted to handle 5% fraud
weights_sub <- ifelse(train_set$Is_Fraud_Num == 1, 5, 1)

cat("\n--- STARTING MARS FITTING (earth) ---\n")
mars_model <- earth(x = train_set[, !(names(train_set) %in% c("Is_Fraud", "Is_Fraud_Num"))], 
                    y = train_set$Is_Fraud_Num, 
                    degree = 2, nk = 40, nprune = 20, 
                    weights = weights_sub, 
                    glm = list(family = binomial),
                    trace = 2)

# Immediate Evaluation
mars_probs <- predict(mars_model, test_set[, !(names(test_set) %in% c("Is_Fraud", "Is_Fraud_Num"))], type = "response")
mars_preds <- factor(ifelse(mars_probs > 0.5, "Fraud", "Normal"), levels = c("Normal", "Fraud"))

cat("\nMARS Confusion Matrix:\n")
print(confusionMatrix(mars_preds, test_set$Is_Fraud, positive = "Fraud")$table)

# Final ROC Curve
roc_mars <- roc(test_set$Is_Fraud_Num, as.vector(mars_probs))
plot(roc_mars, col="firebrick", lwd=3, main=paste("MARS ROC Curve (AUC =", round(auc(roc_mars), 3), ")"))
