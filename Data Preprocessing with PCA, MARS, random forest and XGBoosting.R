# ==========================================
# 1. LOAD LIBRARIES & DATA
# ==========================================
suppressPackageStartupMessages({
  library(caret)
  library(dplyr)
  library(e1071)
  library(corrplot)
  library(pROC)
  library(xgboost)
  library(randomForest)
  library(ggplot2)
})

# Load dataset
# df <- read.csv("new_bank_fraud_detection.csv", stringsAsFactors = FALSE)
# Note: Ensure your CSV is in the working directory

# 1. LOAD DATA (Crucial: Make sure this runs first!)
df <- read.csv("new_bank_fraud_detection.csv", stringsAsFactors = FALSE)

# 2. FEATURE ENGINEERING
# If 'df' is a dataframe, mutate will work. 
# If 'df' is missing, R tries to use the 'df()' function, causing your error.
df_engineered <- df %>%
  mutate(
    Time_Obj = as.POSIXct(Transaction_Time, format="%H:%M:%S"),
    Hour = as.numeric(format(Time_Obj, "%H")),
    Date_Obj = as.Date(Transaction_Date),
    Is_Weekend = ifelse(weekdays(Date_Obj) %in% c("Saturday", "Sunday"), 1, 0),
    Is_Night = ifelse(Hour >= 0 & Hour <= 5, 1, 0),
    Amount_Balance_Ratio = Transaction_Amount / (Account_Balance + 1),
    Log_Residual_Bal = log1p(pmax(0, Account_Balance - Transaction_Amount)),
    Log_Amt = log1p(Transaction_Amount),
    Location_Mismatch = ifelse(State != Transaction_Location, 1, 0),
    Is_Digital = ifelse(Device_Type %in% c("Smartphone", "Laptop"), 1, 0),
    Age_Digital = Age * Is_Digital,
    Amt_Night_Interaction = log1p(Transaction_Amount) * Is_Night
  )
# ==========================================
# 3. TARGET ENCODING
# ==========================================
set.seed(123)
target_cols <- c("State", "Bank_Branch", "Transaction_Location", "Transaction_Description")
m <- 15 
global_mean <- mean(as.numeric(as.factor(df_engineered$Is_Fraud)) - 1)

for (col in target_cols) {
  stats <- df_engineered %>%
    group_by(.data[[col]]) %>%
    summarise(n = n(), local_m = mean(as.numeric(as.factor(Is_Fraud)) - 1)) %>%
    mutate(smoothed = (n * local_m + m * global_mean) / (n + m))
  
  df_engineered[[paste0(col, "_Risk")]] <- stats$smoothed[match(df_engineered[[col]], stats[[col]])]
}

# ==========================================
# 4. CLEANING & DUMMY ENCODING
# ==========================================
drop_cols <- c("Transaction_ID", "Merchant_ID", "X", "Transaction_Time", 
               "Transaction_Date", "Time_Obj", "Date_Obj", target_cols)

df_final <- df_engineered %>% dplyr::select(-all_of(drop_cols))
df_final <- df_final %>% mutate(across(where(is.character), as.factor))

cols_to_keep <- sapply(df_final, function(x) length(unique(x)) > 1)
df_final <- df_final[, cols_to_keep]

fraudClass <- as.factor(df_final$Is_Fraud)
predictors_only <- df_final %>% dplyr::select(-Is_Fraud)

dummies <- dummyVars(~ ., data = predictors_only, fullRank = TRUE)
fraudX <- as.data.frame(predict(dummies, newdata = predictors_only))

# ==========================================
# 5. CARET PREPROCESS
# ==========================================
fraudPP <- preProcess(fraudX, method = c("BoxCox", "center", "scale", "nzv"))
fraudTrans <- predict(fraudPP, fraudX)

cat("\n--- PREPROCESSING COMPLETE ---\n")
cat("Final number of predictors:", ncol(fraudTrans), "\n")

# ==========================================
# 6. CORRELATION CHECK
# ==========================================
fraudCorr <- cor(fraudTrans)
corrplot(fraudCorr, method="color", order="hclust", tl.cex=0.5, 
         main="\n\nCorrelation of Final Features")

# ==========================================
# 7. SKEWNESS & BOXPLOTS (2 WINDOWS ONLY)
# ==========================================
# Calculate skewness for all columns
skews <- apply(fraudTrans, 2, skewness)
cat("\n--- SKEWNESS OF PREDICTORS ---\n")
print(round(skews, 4))

predictor_names <- colnames(fraudTrans)
num_vars <- length(predictor_names)
midpoint <- ceiling(num_vars / 2)

# Define two groups of variables
groups <- list(
  Window1 = predictor_names[1:midpoint],
  Window2 = predictor_names[(midpoint + 1):num_vars]
)

for (i in 1:length(groups)) {
  current_vars <- groups[[i]]
  n_vars <- length(current_vars)
  
  # Launch a new window
  if(.Platform$OS.type == "windows") windows(width=12, height=10) else dev.new()
  
  # Calculate layout grid dynamically based on variable count
  grid_rows <- ceiling(sqrt(n_vars))
  grid_cols <- ceiling(n_vars / grid_rows)
  par(mfrow = c(grid_rows, grid_cols), mar = c(4, 4, 3, 1))
  
  for (var in current_vars) {
    skew_val <- round(skews[var], 2)
    boxplot(fraudTrans[, var] ~ fraudClass, 
            main = paste0(var, "\n(Skew: ", skew_val, ")"),
            col = c("skyblue", "orange"),
            xlab = "Fraud", ylab = "Scaled Value",
            cex.main = 0.7)
  }
}

# ==========================================
# 8. PCA ANALYSIS
# ==========================================
fraud_pca <- prcomp(fraudTrans, center = TRUE, scale. = TRUE)

if(.Platform$OS.type == "windows") windows(width=12, height=6) else dev.new()
par(mfrow=c(1,2))
plot(fraud_pca, type = "l", main = "Scree Plot")
abline(h = 1, col="red", lty=2)

var_exp <- cumsum(fraud_pca$sdev^2) / sum(fraud_pca$sdev^2)
plot(var_exp, type = "b", main = "Cumulative Variance")
abline(h = 0.8, col="darkgreen")

# ==========================================
# 8.1 PCA DETAILED BREAKDOWN (80% Variance)
# ==========================================

# 1. Calculate Variance Explained
pca_summary <- summary(fraud_pca)$importance
var_explained <- pca_summary[2, ]  # Proportion of Variance
cum_var <- pca_summary[3, ]        # Cumulative Proportion

# 2. Identify PCs needed for 80% Variance
num_pcs_80 <- which(cum_var >= 0.80)[1]
cat("\n--- PCA SUMMARY ---\n")
cat("Number of PCs to reach 80% variance:", num_pcs_80, "\n\n")

# 3. Create a table of Variance per PC
pc_table <- data.frame(
  PC = paste0("PC", 1:num_pcs_80),
  Variance_Contribution = paste0(round(var_explained[1:num_pcs_80] * 100, 2), "%"),
  Cumulative_Total = paste0(round(cum_var[1:num_pcs_80] * 100, 2), "%")
)
print(pc_table)

# 4. Extract Predictor Weights (Loadings) for these PCs
# The 'rotation' matrix contains the weights of each original predictor
weights <- as.data.frame(fraud_pca$rotation[, 1:num_pcs_80])

# 5. Find the TOP 5 predictors for each PC based on absolute weight
cat("\n--- TOP PREDICTORS PER PC (By Absolute Weight) ---\n")
for(i in 1:num_pcs_80) {
  pc_col <- weights[, i, drop = FALSE]
  top_preds <- pc_col %>%
    mutate(AbsWeight = abs(.[,1])) %>%
    arrange(desc(AbsWeight)) %>%
    head(5)
  
  cat("\n", colnames(weights)[i], "Top Contributors:\n")
  print(top_preds)
}

# 6. Visualization of Predictor Loadings for the first 2 PCs
if(.Platform$OS.type == "windows") windows(width=10, height=8) else dev.new()
biplot(fraud_pca, cex = 0.7, col = c("gray", "red"),
       main = "Biplot: Predictor Weights on PC1 and PC2")

# ==========================================
# 9. XGBOOST MODELING
# ==========================================
set.seed(123)
trainIndex <- createDataPartition(df_engineered$Is_Fraud, p = 0.8, list = FALSE)
train_data <- df_engineered[trainIndex, ]
test_data  <- df_engineered[-trainIndex, ]

# Matrix Preparation
X_train <- train_data %>% dplyr::select(ends_with("_Risk"), Hour, Is_Weekend, Is_Night, 
                                        Amount_Balance_Ratio, Log_Residual_Bal, Log_Amt, 
                                        Age, Age_Digital, Amt_Night_Interaction) %>% as.matrix()

X_test <- test_data %>% dplyr::select(ends_with("_Risk"), Hour, Is_Weekend, Is_Night, 
                                      Amount_Balance_Ratio, Log_Residual_Bal, Log_Amt, 
                                      Age, Age_Digital, Amt_Night_Interaction) %>% as.matrix()

y_train <- as.numeric(as.factor(train_data$Is_Fraud)) - 1
y_test  <- as.numeric(as.factor(test_data$Is_Fraud)) - 1

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test, label = y_test)

ratio <- sum(y_train == 0) / sum(y_train == 1)
params <- list(objective = "binary:logistic", eval_metric = "auc", eta = 0.01, 
               max_depth = 3, gamma = 5, subsample = 0.5, colsample_bytree = 0.5, 
               scale_pos_weight = ratio)

xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 2000, 
                       evals = list(train = dtrain, test = dtest), 
                       print_every_n = 500, early_stopping_rounds = 100)

# --- XGBoost Confusion Matrix ---
xgb_preds_prob <- predict(xgb_model, dtest)
xgb_preds_class <- factor(ifelse(xgb_preds_prob > 0.5, 1, 0), levels = c(0, 1))
y_test_factor <- factor(y_test, levels = c(0, 1))

cat("\n--- XGBOOST CONFUSION MATRIX ---\n")
print(confusionMatrix(xgb_preds_class, y_test_factor, positive = "1"))

# --- XGBoost Importance Plot ---
importance_matrix <- xgb.importance(feature_names = colnames(X_train), model = xgb_model)

# Open a new window for the plot
if(.Platform$OS.type == "windows") windows(width=10, height=8) else dev.new()
xgb.plot.importance(importance_matrix, main = "XGBoost: Top Predictors")

# ==========================================
# 10. RANDOM FOREST & EVALUATION
# ==========================================
features <- colnames(X_train)
train_balanced <- downSample(x = train_data[, features],
                             y = as.factor(train_data$Is_Fraud),
                             yname = "Is_Fraud")

rf_model <- randomForest(Is_Fraud ~ ., data = train_balanced, ntree = 200, mtry = 3, importance = TRUE)

# Final Diagnostics
rf_probs <- predict(rf_model, newdata = test_data[, features], type = "prob")[,2]
rf_roc <- roc(test_data$Is_Fraud, rf_probs)
cat("\n--- RANDOM FOREST RESULTS ---\n")
cat("Test AUC:", round(rf_roc$auc, 4), "\n")

if(.Platform$OS.type == "windows") windows() else dev.new()
varImpPlot(rf_model, main = "Variable Importance")

# --- Random Forest Confusion Matrix ---
# Ensure rf_preds and test_data$Is_Fraud are both factors with the same levels
rf_preds <- predict(rf_model, newdata = test_data[, features])

y_test_rf <- as.factor(test_data$Is_Fraud)

cat("\n--- RANDOM FOREST CONFUSION MATRIX ---\n")
print(confusionMatrix(rf_preds, y_test_rf, positive = "1"))

# ==========================================
# 12. MARS MODEL (Earth)
#12.1 Downsampling ==========================================
library(earth)

set.seed(123)

# 1. BUILD THE MODEL
# glm = list(family=binomial) tells MARS this is a classification (logistic) task
# degree = 2 allows for interaction terms between predictors
# nprune is the maximum number of terms in the final model (pruning)
mars_model <- earth(
  as.factor(Is_Fraud) ~ ., 
  data = train_balanced, 
  degree = 2,           
  glm = list(family = binomial),
  nprune = 15           
)

# How it's built:
# MARS performs a 'forward pass' creating many hinge functions, 
# then a 'backward pass' (pruning) to remove terms that contribute 
# the least to the model's predictive accuracy.

# 2. PREDICT ON TEST SET
mars_probs <- predict(mars_model, newdata = test_data[, features], type = "response")[,1]
mars_preds <- factor(ifelse(mars_probs > 0.5, 1, 0), levels = c(0, 1))

# 3. CONFUSION MATRIX
cat("\n--- MARS MODEL CONFUSION MATRIX ---\n")
print(confusionMatrix(mars_preds, as.factor(test_data$Is_Fraud), positive = "1"))

# 4. IMPORTANT PREDICTORS
# MARS uses "Variable Importance" based on the reduction in GRSq (Generalized R-squared)
ev <- evimp(mars_model)
if(.Platform$OS.type == "windows") windows(width=10, height=8) else dev.new()
plot(ev, main = "MARS: Variable Importance")

# 5. DRAW ROC CURVE
mars_roc <- roc(test_data$Is_Fraud, mars_probs)
if(.Platform$OS.type == "windows") windows(width=8, height=8) else dev.new()
plot(mars_roc, col = "darkblue", lwd = 3, main = paste("MARS ROC Curve\nAUC:", round(mars_roc$auc, 4)))
abline(a=0, b=1, lty=2, col="gray")

# 12.2 Train Model with Strict Pruning
mars_strict <- earth(
  as.factor(Is_Fraud) ~ ., 
  data = train_balanced, 
  degree = 2,
  glm = list(family = binomial),
  nprune = 8             # Only the top 8 most powerful rules
)

# 12.2 Predict & Evaluate
p_strict <- predict(mars_strict, test_data[, features], type = "response")[,1]
cm_strict <- confusionMatrix(factor(ifelse(p_strict > 0.5, 1, 0), levels=c(0,1)), 
                             as.factor(test_data$Is_Fraud), positive = "1")
roc_strict <- roc(test_data$Is_Fraud, p_strict)

# 12.3 Train Model with Higher Penalty
mars_penalty <- earth(
  as.factor(Is_Fraud) ~ ., 
  data = train_balanced, 
  degree = 2,
  penalty = 4,           # Increased from default to suppress noise
  glm = list(family = binomial),
  nprune = 15
)

# 12.3 Predict & Evaluate
p_penalty <- predict(mars_penalty, test_data[, features], type = "response")[,1]
cm_penalty <- confusionMatrix(factor(ifelse(p_penalty > 0.5, 1, 0), levels=c(0,1)), 
                              as.factor(test_data$Is_Fraud), positive = "1")
roc_penalty <- roc(test_data$Is_Fraud, p_penalty)

# 12.4 Train Model with 10-Fold Cross-Validation
mars_cv <- earth(
  as.factor(Is_Fraud) ~ ., 
  data = train_balanced, 
  degree = 2,
  nfold = 10,            # Cross-validation to find consistent signals
  pmethod = "cv",        # Use CV to prune the model
  glm = list(family = binomial),
  nprune = 15
)

# 12.4 Predict & Evaluate
p_cv <- predict(mars_cv, test_data[, features], type = "response")[,1]
cm_cv <- confusionMatrix(factor(ifelse(p_cv > 0.5, 1, 0), levels=c(0,1)), 
                         as.factor(test_data$Is_Fraud), positive = "1")
roc_cv <- roc(test_data$Is_Fraud, p_cv)

# 12.5 Train Model with Span Control
mars_span <- earth(
  as.factor(Is_Fraud) ~ ., 
  data = train_balanced, 
  degree = 2,
  minspan = 20,          # Minimum 20 observations between knots
  glm = list(family = binomial),
  nprune = 15
)

# 12.5 Predict & Evaluate
p_span <- predict(mars_span, test_data[, features], type = "response")[,1]
cm_span <- confusionMatrix(factor(ifelse(p_span > 0.5, 1, 0), levels=c(0,1)), 
                           as.factor(test_data$Is_Fraud), positive = "1")
roc_span <- roc(test_data$Is_Fraud, p_span)

# 12.6 Comparison Table
comparison <- data.frame(
  Strategy = c("Penalty (Noise Reduc)", "CV (Signal Boost)", "Span (Stability)", "Strict (Clarity)"),
  AUC = c(auc(roc_penalty), auc(roc_cv), auc(roc_span), auc(roc_strict)),
  Sensitivity = c(cm_penalty$byClass["Sensitivity"], cm_cv$byClass["Sensitivity"], 
                  cm_span$byClass["Sensitivity"], cm_strict$byClass["Sensitivity"]),
  F1 = c(cm_penalty$byClass["F1"], cm_cv$byClass["F1"], 
         cm_span$byClass["F1"], cm_strict$byClass["F1"])
)
print(comparison)

# 2. ROC Plot Comparison
if(.Platform$OS.type == "windows") windows(width=10, height=8) else dev.new()
plot(roc_penalty, col="red", lwd=2, main="MARS Tuning Strategies Comparison")
plot(roc_cv, col="blue", lwd=2, add=TRUE)
plot(roc_span, col="green", lwd=2, add=TRUE)
plot(roc_strict, col="purple", lwd=2, add=TRUE)
legend("bottomright", legend=comparison$Strategy, col=c("red", "blue", "green", "purple"), lwd=2)


