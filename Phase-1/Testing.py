from sqlalchemy import create_engine
engine = create_engine("postgresql+psycopg2://postgres:Teza_281200!@localhost:5432/energy_intelligence")
conn = engine.connect()
print(conn)


from db import engine
print(engine)

import gridstatus
ny = gridstatus.NYISO()
df = ny.get_lmp(date="today", market="DAY_AHEAD_HOURLY")
print(df.head())


