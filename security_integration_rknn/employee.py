import os
import pandas as pd
import sqlite3
import io

def upload_employee_excel_to_db_from_bytes(file_content):
    expected_columns = [
        "Emp ID",
        "Name",
        "Department",
        "Function / Employee Group",
        "Employee Email",
        "Supervisor Email",
        "Mobile No",
        "Vehicle No"
    ]

    db_name = "employee_master.db"

    try:
        # Read Excel from bytes
        df = pd.read_excel(io.BytesIO(file_content))

        # Check columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in expected_columns]
        if missing_cols:
            return {"status": "error", "message": f"Missing columns: {missing_cols}"}
        if extra_cols:
            return {"status": "error", "message": f"Unexpected columns in Excel: {extra_cols}"}

        # Clean empty rows
        df.dropna(how="all", inplace=True)

        # Overwrite DB file
        if os.path.exists(db_name):
            os.remove(db_name)

        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Create table
        columns_sql = ", ".join([f'"{col}" TEXT' for col in expected_columns])
        cursor.execute(f'CREATE TABLE employee_master ({columns_sql})')

        # Insert data
        df.to_sql("employee_master", conn, if_exists="append", index=False)

        conn.commit()
        conn.close()

        return {"status": "success", "message": "Data imported to employee_master.db"}

    except Exception as e:
        return {"status": "error", "message": str(e)}