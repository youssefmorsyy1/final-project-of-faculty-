from sqlalchemy import create_engine
import pandas as pd

DB_URL = "postgresql://postgres:111222@localhost:5432/soccer_db"

engine = create_engine(DB_URL)

OUTPUT_FILE = "postgres_full_database_report.txt"

with open(OUTPUT_FILE, "w", encoding="utf-8") as report:

    def write(text=""):
        report.write(str(text) + "\n")

    # =========================================
    # GET TABLES
    # =========================================
    tables_query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_type = 'BASE TABLE'
    ORDER BY table_name;
    """

    tables = pd.read_sql(tables_query, engine)["table_name"].tolist()

    write("=" * 80)
    write("POSTGRESQL FULL DATABASE TEXT REPORT")
    write("=" * 80)

    # =========================================
    # LOOP TABLES
    # =========================================
    for table in tables:

        write("\n" + "#" * 80)
        write(f"TABLE: {table}")
        write("#" * 80)

        # Row count
        row_count = pd.read_sql(
            f'SELECT COUNT(*) AS cnt FROM "{table}"',
            engine
        )["cnt"][0]

        write(f"\nRow Count: {row_count}")

        # Schema
        schema_query = f"""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = '{table}'
        ORDER BY ordinal_position;
        """

        schema = pd.read_sql(schema_query, engine)

        write(f"Column Count: {len(schema)}")
        write("\n---------------- COLUMN DETAILS ----------------")

        for _, col in schema.iterrows():

            column = col["column_name"]
            dtype = col["data_type"]

            write(f"\n→ Column: {column}")
            write(f"  Type: {dtype}")
            write(f"  Nullable: {col['is_nullable']}")

            # NULL COUNT
            null_count = pd.read_sql(
                f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" IS NULL',
                engine
            ).iloc[0, 0]

            null_pct = (null_count / row_count * 100) if row_count else 0

            write(f"  Null Count: {null_count}")
            write(f"  Null %: {null_pct:.2f}%")

            # DISTINCT COUNT
            distinct_count = pd.read_sql(
                f'SELECT COUNT(DISTINCT "{column}") FROM "{table}"',
                engine
            ).iloc[0, 0]

            write(f"  Distinct Count: {distinct_count}")

            # SAMPLE VALUES
            sample_df = pd.read_sql(
                f'''
                SELECT "{column}"
                FROM "{table}"
                WHERE "{column}" IS NOT NULL
                LIMIT 5
                ''',
                engine
            )

            samples = sample_df[column].dropna().tolist()
            write(f"  Sample Values: {samples}")

            # NUMERIC STATS
            if dtype in [
                "integer",
                "bigint",
                "smallint",
                "numeric",
                "real",
                "double precision"
            ]:

                stats = pd.read_sql(
                    f'''
                    SELECT
                        MIN("{column}") AS min_val,
                        MAX("{column}") AS max_val,
                        AVG("{column}") AS avg_val
                    FROM "{table}"
                    ''',
                    engine
                )

                write(f"  Min: {stats['min_val'][0]}")
                write(f"  Max: {stats['max_val'][0]}")
                write(f"  Avg: {stats['avg_val'][0]}")

            # DATE/TIME STATS
            elif "date" in dtype or "timestamp" in dtype:

                stats = pd.read_sql(
                    f'''
                    SELECT
                        MIN("{column}") AS min_val,
                        MAX("{column}") AS max_val
                    FROM "{table}"
                    ''',
                    engine
                )

                write(f"  Min Date: {stats['min_val'][0]}")
                write(f"  Max Date: {stats['max_val'][0]}")

        write("\n--------------------------------------------")
        write(f"END OF TABLE: {table}")
        write("--------------------------------------------")

    write("\nDONE")

print(f"Report saved to: {OUTPUT_FILE}")