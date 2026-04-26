# 1. LOAD LIBRARIES
library(AppliedPredictiveModeling)
library(caret)
library(e1071)
library(corrplot)
library(plotmo)
library(earth)
library(dplyr)

# 2. LOAD AND INITIAL CLEANING
df <- read.csv("new_bank_fraud_detection.csv", stringsAsFactors = FALSE)

# Convert Time to Hour
df$Transaction_Time <- as.POSIXct(df$Transaction_Time, format = "%H:%M:%S")
df$Transaction_Hour <- as.numeric(format(df$Transaction_Time, "%H"))
df$Transaction_Time <- NULL

# Convert Date to Weekday
df$Transaction_Date <- as.Date(as.character(df$Transaction_Date), format="%d-%m-%Y")
df$Transaction_Weekday <- as.factor(weekdays(df$Transaction_Date))
df$Transaction_Date <- NULL

# Remove Identifiers (Unique strings that don't help modeling)
df <- df[, !(names(df) %in% c("Transaction_ID", "Merchant_ID", "X", "Transaction_Currency"))]

# Binary Encoding for Gender
df$Gender <- ifelse(df$Gender == "Female", 1, 0)

# 3. HANDLE HIGH CARDINALITY (Target Encoding)
# We replace names with the average fraud probability for that category
# This prevents creating 500+ dummy columns.
high_card_vars <- c("State", "Bank_Branch", "Transaction_Location", "Transaction_Description")

for (col in high_card_vars) {
  # Calculate mean fraud rate per category
  target_means <- df %>%
    group_by(across(all_of(col))) %>%
    summarise(mean_target = mean(as.numeric(as.character(Is_Fraud)), na.rm = TRUE), .groups = 'drop')
  
  # Map those means back to the dataframe as new numeric columns
  df[[paste0(col, "_Encoded")]] <- target_means$mean_target[match(df[[col]], target_means[[col]])]
}

# Remove original high cardinality text columns
df_clean <- df[, !(names(df) %in% high_card_vars)]

# 4. PREPROCESS NUMERIC DATA (BoxCox, Center, Scale)
# Identify numeric columns (excluding the target Is_Fraud)
numeric_cols <- names(df_clean)[sapply(df_clean, is.numeric) & names(df_clean) != "Is_Fraud"]

# Apply transformations to stabilize variance and skewness
pp_processor <- preProcess(df_clean[, numeric_cols], method = c("BoxCox", "center", "scale"))
df_clean[, numeric_cols] <- predict(pp_processor, df_clean[, numeric_cols])

# --- 5. DUMMY ENCODING (Corrected) ---
# Ensure the target is saved separately
final_y <- as.factor(df_clean$Is_Fraud)

# Create a predictors-only dataframe
predictors_only <- df_clean[, names(df_clean) != "Is_Fraud"]

# Use ~ . with NO variable on the left side of the ~
# This tells caret there is no response variable to look for
dummies_model <- dummyVars(~ ., data = predictors_only)

# Now predict will work perfectly without looking for 'Is_Fraud'
final_x <- as.data.frame(predict(dummies_model, newdata = predictors_only))

# Check dimensions to be sure
dim(final_x)

# 6. TRAIN / TEST SPLIT
set.seed(123)
trainIndex <- createDataPartition(final_y, p = 0.8, list = FALSE)
trainX <- final_x[trainIndex, ]
trainY <- final_y[trainIndex]
testX  <- final_x[-trainIndex, ]
testY  <- final_y[-trainIndex]

# 7. FIT MARS MODEL 
# REVISED MODELING (Subsampling to prevent freezing) ---

# Create a balanced training set: Keep all Fraud, but only 20k Non-Fraud
set.seed(123)
fraud_idx <- which(trainY == 1)
non_fraud_idx <- sample(which(trainY == 0), 20000) # Reduced for speed
sub_idx <- c(fraud_idx, non_fraud_idx)

trainX_sub <- trainX[sub_idx, ]
trainY_sub <- trainY[sub_idx]

# Recalculate weights for the smaller set
# Since we downsampled non-fraud, a 1:1 or 1:5 weight is usually enough now
weights_sub <- ifelse(trainY_sub == 1, 5, 1)

cat("Starting MARS model fitting on", nrow(trainX_sub), "rows...\n")

mars_model <- earth(x = trainX_sub, 
                    y = trainY_sub, 
                    degree = 2,           # Interactions allowed
                    nk = 40,              # Limit max terms in forward pass
                    nprune = 20,          # Final model size
                    weights = weights_sub, 
                    glm = list(family = binomial),
                    trace = 2)            # TRACE = 2 shows you progress so you know it's not stuck

cat("Model fitting complete!\n")      # Force the model to select up to 20 terms

# Now check importance again
print(evimp(mars_model))

# 8. RESULTS AND EVALUATION (Recalibrated)
R
# --- 8. RESULTS AND EVALUATION (Recalibrated for Fraud) ---

# 1. Get raw probabilities
probs <- predict(mars_model, newdata = testX, type = "response")

# 2. Find the "Fraud Signal" cutoff
# Since actual fraud is ~5%, we flag the top 5% most suspicious cases
optimal_threshold <- quantile(probs, 0.95) 
cat("The recalibrated threshold is:", optimal_threshold, "\n")

# 3. Apply the threshold 
# CRITICAL: We ensure '1' is the second level so R recognizes it
predicted_class <- factor(ifelse(probs > optimal_threshold, 1, 0), levels = c("0", "1"))
actual_class <- factor(testY, levels = c("0", "1"))

# 4. Run Confusion Matrix - EXPLICITLY set positive = "1"
# This forces Sensitivity to represent the 'Fraud Catch Rate'
conf_matrix <- confusionMatrix(predicted_class, actual_class, positive = "1")

# 5. Print the results
print(conf_matrix)

# 9. PERFORMANCE VISUALIZATION (ROC Curve)
# install.packages("pROC")
library(pROC)

# Calculate the Area Under the Curve (AUC)
# This tells you how well MARS separates classes regardless of the threshold
roc_score <- roc(testY, as.numeric(probs))
plot(roc_score, main=paste("MARS ROC Curve - AUC:", round(auc(roc_score), 3)), col="blue", lwd=2)
abline(a=0, b=1, lty=2, col="red") # Random chance line

# 1. Load the required libraries
#library(earth)
#library(ggplot2)

# 2. Get importance from the earth-specific function
# We use the 'earth::' prefix to be 100% sure R finds it
ev <- earth::evimp(mars_model)

# 3. Convert the 'gcv' (Generalized Cross-Validation) column to a data frame
# GCV is the standard way to measure importance in MARS
imp_df <- data.frame(
  Feature = rownames(ev),
  Importance = as.numeric(ev[, "gcv"])
)

# 4. Filter out variables with 0 importance
imp_df <- imp_df[imp_df$Importance > 0, ]

# 5. Sort the data frame by Importance
imp_df <- imp_df[order(-imp_df$Importance), ]

# 6. Create the Plot
ggplot(imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "MARS Variable Importance (GCV)",
    subtitle = "Based on earth model selection",
    x = "Predictors",
    y = "Importance Score"
  )
