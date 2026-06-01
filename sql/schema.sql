-- Represents the SQLite structure created from the CSV
CREATE TABLE loan_applications (
    SK_ID_CURR INTEGER PRIMARY KEY,
    TARGET INTEGER,
    AMT_INCOME_TOTAL REAL,
    AMT_CREDIT REAL,
    DAYS_BIRTH INTEGER,
    DAYS_EMPLOYED INTEGER
);