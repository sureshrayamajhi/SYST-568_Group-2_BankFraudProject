# install.packages("AppliedPredictiveModeling")
library(AppliedPredictiveModeling)

# install.packages("caret")
library(caret)

# install.packages("e1071")
library(e1071)

# install.packages("corrplot")
library(corrplot)

#install.packages("plotmo")
library(plotmo)

#install.packages("earth")
library(earth)

#load dataset
df <- read.csv("new_bank_fraud_detection.csv", stringsAsFactors = FALSE)
str(df)


# Replace Transaction_tIME With Transaction_Hour
# Convert to time format
df$Transaction_Time <- as.POSIXct(df$Transaction_Time, format = "%H:%M:%S")

# Create hour variable
df$Transaction_Hour <- as.numeric(format(df$Transaction_Time, "%H"))

# Drop original column
df$Transaction_Time <- NULL


#Replace Transaction_date with Transaction_Weekday
# Ensure Transaction_Date is character
df$Transaction_Date <- as.character(df$Transaction_Date)

# Convert to Date format (assuming format is "dd-mm-yyyy")
df$Transaction_Date <- as.Date(df$Transaction_Date, format="%d-%m-%Y")

# Extract weekday (Monday, Tuesday, ...)
df$Transaction_Weekday <- weekdays(df$Transaction_Date)

# Optional: convert to factor for modeling or plots
df$Transaction_Weekday <- as.factor(df$Transaction_Weekday)

# Remove original Transaction_Date column
df$Transaction_Date <- NULL

# Check the result
head(df[, c("Transaction_Weekday")])
summary(df$Transaction_Weekday)

# Verify structure
str(df)

#Remove identifier column
df <- df[, !(names(df) %in% c("Transaction_ID", "Merchant_ID", "X","Transaction_Currency"))]
str(df)
#convert variables Properly
# Convert Gender to dummy
df$Gender <- ifelse(df$Gender == "Female", 1, 0)

# Load necessary packages
library(dplyr) # For data manipulation

# Assuming 'df' is your preprocessed data frame

# Identify categorical columns (excluding Is_Fraud itself)
# We can identify character and factor columns as categorical
categorical_cols <- names(df)[sapply(df, function(x) is.character(x) | is.factor(x))]
categorical_cols <- setdiff(categorical_cols, c("Is_Fraud")) # Exclude the target variable

# Loop through each categorical column and generate statistics
for (col_name in categorical_cols) {
  cat("\n--- Statistics for:", col_name, "---\n")
  
  df %>%
    group_by( across(col_name), Is_Fraud) %>%
    summarise(Count = n(), .groups = 'drop') %>%
    group_by( across(col_name)) %>%
    mutate(Percentage = Count / sum(Count) * 100) %>%
    arrange(desc(Count)) %>%
    print()
}

# Additionally, you might want to see how each category within a variable
# contributes to the overall fraud cases. This provides a different perspective.
cat("\n--- Fraud Percentage by Category ---\n")
for (col_name in categorical_cols) {
  cat("\n--- Fraud Percentage for:", col_name, "---\n")
  
  df %>%
    group_by(across(col_name)) %>%
    summarise(
      Total_Transactions = n(),
      Fraud_Transactions = sum(as.numeric(as.character(Is_Fraud))), # Convert factor to numeric (0, 1) then sum
      Fraud_Percentage = (Fraud_Transactions / Total_Transactions) * 100,
      .groups = 'drop'
    ) %>%
    arrange(desc(Fraud_Percentage)) %>%
    print()
}


# Convert class to factor
df$Is_Fraud <- as.factor(df$Is_Fraud)

#convert other categorical variables to factor
factorVars <- c("State", "Bank_Branch", "Account_Type",
                "Transaction_Type", "Merchant_Category",
                "Transaction_Device", "Device_Type",
                "Transaction_Currency","Transaction_Location",
                "Transaction_Description",
                "Transaction_Weekday")

df[factorVars] <- lapply(df[factorVars], as.factor)

#separate predictors and class
fraudX <- df[, !(names(df) %in% c("Is_Fraud"))]
fraudClass <- df$Is_Fraud

#For correlation and predictors we need numeric predictors only
fraudX_num <- fraudX[, sapply(fraudX, is.numeric)]

#check class balance
table(fraudClass)
prop.table(table(fraudClass))

#check skewness
apply(fraudX_num, 2, skewness)

#Example for one variable
skewness(fraudX_num$Transaction_Amount)

#Using Caret's Preprocess for skewness
#Remove Zero variance predictors first
isZV <- apply(fraudX_num, 2, function(x) length(unique(x)) == 1)

fraudX_num <- fraudX_num[, !isZV]

#Apply Boxcox+center+scale
fraudPP <- preProcess(fraudX_num, method = c("BoxCox", "center", "scale"))

fraudTrans <- predict(fraudPP, fraudX_num)

#Histogram before and after transformation
#Example Transaction amount
par(mfrow=c(1,2))

hist(fraudX_num$Transaction_Amount,
     main="Original",
     xlab="Transaction Amount")

hist(fraudTrans$Transaction_Amount,
     main="Transformed",
     xlab="Transformed Amount")

#Check skewness
skewness(fraudX_num$Transaction_Amount)
skewness(fraudTrans$Transaction_Amount)

#PCA on all predictors
fraudPCA <- prcomp(fraudTrans, center=TRUE, scale.=TRUE)

percentVariance = fraudPCA$sd^2/sum(fraudPCA$sd^2)*100
percentVariance[1:5]

#Plot Variance
plot(percentVariance,
     type="l",
     xlab="Component",
     ylab="Percentage of Variance",
     main="PCA - Fraud Dataset")


#Looking at transformed values
head(fraudPCA$x[,1:4])

#Correlation matrix
fraudCorr <- cor(fraudTrans)

#corrplot(fraudCorr, order="hclust", tl.cex=.6)
#Fix for error issue for above script
if(ncol(filteredFraudData) >= 2){
  corrplot(cor(filteredFraudData), order="hclust", tl.cex=.6)
} else {
  cat("Not enough variables to plot correlation after removing high correlations.\n")
}


#remove highly correlated predictors
highCorr <- findCorrelation(fraudCorr, cutoff=0.75)

filteredFraudData <- fraudTrans[, -highCorr]

corrplot(cor(filteredFraudData), order="hclust", tl.cex=.6)

#correlation with class label
cor(fraudTrans, as.numeric(fraudClass))

#Boxplot of predictors vs Fraud
par(mfrow=c(1,4))

boxplot(fraudX_num$Transaction_Amount ~ fraudClass,
        main="Transaction Amount")

boxplot(fraudX_num$Account_Balance ~ fraudClass,
        main="Account Balance")

boxplot(fraudX_num$Age ~ fraudClass,
        main="Age")

boxplot(fraudX_num$Transaction_Hour ~ fraudClass,
        main="Transaction_Hour")

#Reset
par(mfrow=c(1,1))

#Histogram of skewed predictors
par(mfrow=c(1,4))
hist(fraudX_num$Transaction_Amount)
hist(fraudX_num$Account_Balance)
hist(fraudX_num$Age)
hist(fraudX_num$Transaction_Hour)

#Histogram for Fraud Cases Only
# Subset numeric variables for fraud only
fraudX_num_only <- fraudX_num[fraudClass == 1, ]  # fraudClass == 1 means fraud

# Plot histograms for only fraud cases
par(mfrow=c(1,4), mar=c(4,4,2,1))  # adjust margins if needed

hist(fraudX_num_only$Transaction_Amount,
     main="Transaction Amount (Fraud Only)",
     xlab="Transaction Amount",
     col="red",
     breaks=30)

hist(fraudX_num_only$Account_Balance,
     main="Account Balance (Fraud Only)",
     xlab="Account Balance",
     col="red",
     breaks=30)

hist(fraudX_num_only$Age,
     main="Age (Fraud Only)",
     xlab="Age",
     col="red",
     breaks=30)

hist(fraudX_num_only$Transaction_Hour,
     main="Transaction Hour (Fraud Only)",
     xlab="Transaction Hour",
     col="red",
     breaks=24)  # use 24 bins for hours

#Histogram for Non-fraud cases only
# Subset numeric variables for non-fraud only
nonFraudX_num_only <- fraudX_num[fraudClass == 0, ]  # fraudClass == 0 means non-fraud

# Plot histograms for only non-fraud cases
par(mfrow=c(1,4), mar=c(4,4,2,1))  # adjust margins if needed

hist(nonFraudX_num_only$Transaction_Amount,
     main="Transaction Amount (Non-Fraud Only)",
     xlab="Transaction Amount",
     col="blue",
     breaks=30)

hist(nonFraudX_num_only$Account_Balance,
     main="Account Balance (Non-Fraud Only)",
     xlab="Account Balance",
     col="blue",
     breaks=30)

hist(nonFraudX_num_only$Age,
     main="Age (Non-Fraud Only)",
     xlab="Age",
     col="blue",
     breaks=30)

hist(nonFraudX_num_only$Transaction_Hour,
     main="Transaction Hour (Non-Fraud Only)",
     xlab="Transaction Hour",
     col="blue",
     breaks=24)  # use 24 bins for hours

# Function to plot a categorical variable only if it has 5 or 6 unique levels


