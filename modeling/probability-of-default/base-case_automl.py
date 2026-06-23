import sklearn.datasets
import sklearn.model_selection
import autosklearn.classification
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

# Load dataset
feature_cols = ['CreditScoreRangeLower', 'CreditScoreRangeUpper', 'FirstRecordedCreditLine', 'CurrentCreditLines', 'OpenCreditLines', 'TotalCreditLinespast7years', 'TotalTrades',
                'OpenRevolvingAccounts', 'OpenRevolvingMonthlyPayment', 'RevolvingCreditBalance', 'BankcardUtilization', 'AvailableBankcardCredit',
                'CurrentDelinquencies', 'AmountDelinquent', 'DelinquenciesLast7Years', 'TradesNeverDelinquent', 'PublicRecordsLast10Years', 'PublicRecordsLast12Months',
                'InquiriesLast6Months', 'TotalInquiries', 'TradesOpenedLast6Months', 'DebtToIncomeRatio', 'EmploymentStatus', 'EmploymentStatusDuration', 
                'Occupation', 'StatedMonthlyIncome', 'IncomeRange', 'IsBorrowerHomeowner', 'Term', 'LoanOriginalAmount', 'ListingCategory', 'BorrowerState', 'CurrentlyInGroup']

X, y = sklearn.datasets.load_svmlight_file("data/processed/train_data.csv")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

# Configure the AutoML Classifier
automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=1800, # 30 minutes for the entire AutoML process
                                                          per_run_time_limit=300, # 5 minutes per run
                                                          n_jobs=-1) 

automl.fit(X_train, y_train)

predictions = automl.predict(X_test)

# 1. Calculate Discriminatory Metrics
auc = roc_auc_score(y_true, y_prob)
gini = 2 * auc - 1

# KS Statistic
preds_default = y_prob[y_true == 1]
preds_non_default = y_prob[y_true == 0]
ks_stat, _ = ks_2samp(preds_default, preds_non_default)

# 2. Calculate Calibration Metrics
brier = brier_score_loss(y_true, y_prob)
loss = log_loss(y_true, y_prob)