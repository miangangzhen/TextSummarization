import json
import tensorflow as tf
import pandas as pd


def csv2json(in_path, out_path):
    with open(out_path, "w", encoding="utf-8") as fo:
        for file in tf.gfile.Glob(in_path):
            df = pd.read_csv(file, encoding="utf-8")
            df["content"] = df["content"].astype(str)
            df["title"] = df["title"].astype(str)
            for _, row in df.iterrows():
                string = json.dumps({"content": row["content"], "title": row["title"]})
                fo.write(string + "\n")


if __name__ == "__main__":
    in_path = "F:/chinese_summarization/predict/*.csv"
    out_path = "F:/chinese_summarization/predict/predict.json"
    csv2json(in_path, out_path)
