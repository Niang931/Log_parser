from fastapi import FastAPI
from .connect import  get_cursor

app = FastAPI()

@app.post("/tables")
def create_table(table_name:str):
    with get_cursor(dict_cursor=True) as cur:
        cur.execute(f""""
                    f"CREATE TABLE app.{table_name}"
                    f"(id SERIAL PRIMARY KEY,"
                    f"data TEXT"
                    f")""")

        cur.execute("""INSERT INTO app.registry (table_name)"
                    "VALUES (%s)""", (table_name))


@app.get("/tables")
def get_tables():
    with get_cursor(dict_cursor=True) as cur:
        cur.execute("""
        SELECT * FROM app.registry""")

        rows = cur.fetchall()

    return [
        {
            "id":r[0],
            "table_name":r[1],
            "date_created":r[2]
        }
        for r in rows
    ]


@app.post("/tables/{table_name}/data")
def insert_data(table_name:str, data:str):
    with get_cursor(dict_cursor=True) as cur:
        cur.execute(f"""
            INSERT INTO app.{table_name} (data)
            VALUES (%s)""", (data,))

@app.get("/tables/{table_name}/data")
def get_data(table_name:str):
    with get_cursor(dict_cursor=True) as cur:
        cur.execute(f"""SELECT * FROM app.{table_name}""")
        rows = cur.fetchall()
    return rows