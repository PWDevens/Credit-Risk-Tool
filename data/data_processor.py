import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Define file paths
    raw_data_path = os.path.join("raw", "prosperLoanData.csv")
    train_output_path = os.path.join("processed", "train_data.csv")
    test_output_path = os.path.join("processed", "test_data.csv")

    # Load dataset
    if not os.path.exists(raw_data_path):
        print(f"Raw data file not found at {raw_data_path}. Please ensure the dataset is downloaded; see builder.py for information to load data.")
        return
    df = pd.read_csv(raw_data_path)
    
    # Handle missing values in target column
    df = df.dropna(subset=['LoanStatus'])

    # Creating binary of target column to predict default
    """
    LoanStatus is the current status of the loan in categorical values:
    - Cancelled
    - Chargedoff
    - Completed
    - Current
    - Defaulted
    - FinalPaymentInProgress
    - PastDue. 
    
    Note: PastDue status will be accompanied by a delinquency bucket. 
    """
    df['LoanStatus'] = df['LoanStatus'].apply(lambda x: 1 if x in ['Defaulted', 'Chargedoff'] else 0)

    # Split the dataset into training and testing sets
    # Stratifying the binary target to keep default rates equal in train/test 
    train_df, test_df = train_test_split(df, 
                                         test_size=0.2, 
                                         random_state=42,
                                         stratify=df['LoanStatus'])
    
    # Save outputs
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

if __name__ == "__main__":
    main()