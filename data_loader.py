from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class FAQEntry:
    question: str
    answer: str


def load_csv_data(file_path: str) -> list[FAQEntry]:
    csv_path = Path(file_path)
    df = pd.read_csv(csv_path)

    required_columns = {"question", "answer"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV file is missing required columns: {missing}")

    cleaned_df = (
        df.loc[:, ["question", "answer"]]
        .dropna()
        .astype(str)
        .apply(lambda column: column.str.strip())
    )
    cleaned_df = cleaned_df[
        (cleaned_df["question"] != "") & (cleaned_df["answer"] != "")
    ]
    cleaned_df = cleaned_df[
        ~(
            (cleaned_df["question"].str.lower() == "question")
            & (cleaned_df["answer"].str.lower() == "answer")
        )
    ]

    return [
        FAQEntry(question=row.question, answer=row.answer)
        for row in cleaned_df.itertuples(index=False)
    ]
