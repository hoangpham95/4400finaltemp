from asm_proc import ASMProcessor
from csv_proc import CSVProcessor
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == "__main__":
    train_dir = "/home/hoang/Downloads/train"
    csv = CSVProcessor("data", "trainLabels.csv")
    hashes, labels = csv.process()

    X_small, _, y_small, _ = train_test_split(
        hashes, labels, test_size=0.4, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X_small, y_small, test_size=0.2, random_state=42)

    top_features = [1000, 1500, 3000]
    asm = ASMProcessor(train_dir, X_train, y_train)
    df = asm.process()
    df.to_csv("data/train_full.csv")

    for t in top_features:
        print("[+] Processing with", t, "top features")
        top_df = asm.get_top_features(df, t)

        top_df.to_csv("data/train" + str(t) + ".csv")

        top = list(top_df.drop(columns=['file']).columns)

        asm_test = ASMProcessor(train_dir, X_test, y_test,
                                train=False, top_features=top)
        test = asm_test.process_test()
        test.to_csv("data/test" + str(t) + ".csv")
