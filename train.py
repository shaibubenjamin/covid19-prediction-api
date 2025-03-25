import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load dataset and run preprocessing
def load_data():
    """Load and clean data with binary encoding"""
    df = pd.read_excel("covid19.xlsx")
    
    # Drop high-missing columns (>80%)
    missing_pct = (df.isnull().sum() / len(df)) * 100
    df = df.drop(columns=missing_pct[missing_pct > 80].index)
    
    # Create age from birth year
    df['Age'] = pd.Timestamp.now().year - df['Birth Year']
    df['Age'] = df['Age'].fillna(df['Age'].median()).astype(int)
    df = df.drop(columns=['Birth Year'])
    
    #drop columns with NAN
    df = df.dropna().reset_index(drop=True)
    
    # Drop all pending rows
    drop_pending = df[df['Result'] == "PENDING"].index
    df = df.drop(index=drop_pending)
    
    # Convert all YES/NO columns to 1/0
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].isin(['YES', 'NO']).any():
                df[col] = df[col].map({'YES': 1, 'NO': 0, 'UNKNOWN': 0})
            elif col == 'Sex':
                df[col] = df[col].map({'MALE': 1, 'FEMALE': 0, 'OTHER': 0, 'UNKNOWN': 0})
            elif col == 'Result':
                df[col] = df[col].map({'POSITIVE': 1, 'NEGATIVE': 0})
    
    return df

def main():
    print("Loading Data...")
    df = load_data()
    
    # Split data
    X = df.drop(columns=['Result'])
    y = df['Result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    
    print("Training COVID-19 predictor...")
    # train model
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate
    print("Evaluating model...")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    # Save artifacts
    print("Saving model with binary encoded features...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(list(X.columns), 'features.pkl')

    print("Model saved with binary encoded features")

if __name__ == "__main__":
    main()