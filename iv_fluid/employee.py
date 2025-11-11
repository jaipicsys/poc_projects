import os
import pandas as pd
import sqlite3
import io

def upload_patient_excel_to_db_from_bytes(file_content):
    """
    Uploads patient master Excel data into employee_master.db (used for IV monitoring).
    Overwrites the old DB each time new master data is uploaded.
    """

    expected_columns = [
        "Patient ID",
        "Patient Name",
        "Bed Number",
        "Ward",
        "Doctor Name",
        "Doctor Email",
        "Fluid Type",
        "Flow Rate",
        "Time Left",
        "Prescribed Volume"
    ]

    db_name = "employee_master.db"

    try:
        # Read Excel from bytes
        df = pd.read_excel(io.BytesIO(file_content))

        # Check required columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in expected_columns]

        if missing_cols:
            return {"status": "error", "message": f"Missing columns: {missing_cols}"}
        if extra_cols:
            return {"status": "error", "message": f"Unexpected columns: {extra_cols}"}

        # Clean up
        df.dropna(how="all", inplace=True)

        # Delete existing DB (reset each upload)
        if os.path.exists(db_name):
            os.remove(db_name)

        # Create database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Create table schema
        columns_sql = ", ".join([f'"{col}" TEXT' for col in expected_columns])
        cursor.execute(f"CREATE TABLE employee_master ({columns_sql})")

        # Add camera tracking status column
        cursor.execute("ALTER TABLE employee_master ADD COLUMN status TEXT DEFAULT 'stopped'")

        # Insert data
        df.to_sql("employee_master", conn, if_exists="append", index=False)

        conn.commit()
        conn.close()

        return {"status": "success", "message": "Patient data imported into employee_master.db"}

    except Exception as e:
        return {"status": "error", "message": str(e)}
