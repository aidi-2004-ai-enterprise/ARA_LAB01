import seaborn as sns
from sklearn.model_selection import train_test_split

def load_penguins():
    df = sns.load_dataset("penguins")
    print(df.head())
    return df.dropna()

def split_dataset(df):
    """ Split features and target """
    # Features 
    X = df.drop("species", axis=1)
    y = df["species"]
    # train-test split
    return train_test_split(X, y, test_size=0.2, random_state=42)

def main():
    # Load and split the data
    df = load_penguins()
    X_train, X_test, y_train, y_test = split_dataset(df)
    
    # Print the shapes
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)

if __name__ == "__main__":
    main()
