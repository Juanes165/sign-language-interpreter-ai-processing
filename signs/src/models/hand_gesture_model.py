import os
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, inspect, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuracion BD
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# Definicion de nombres de esquemas y tablas
schema_name = os.getenv('SCHEMA')
table_name = os.getenv('TABLE_NAME')

# Comprobar si el esquema existe y crearlo si no
with engine.connect() as connection:
    result = connection.execute(text(f"SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{schema_name}';"))
    if not result.fetchone():
        connection.execute(text(f"CREATE SCHEMA {schema_name};"))
        logger.info(f"Esquema '{schema_name}' creado en la BD.")

# Definicion de la tabla hands_gestures
class HandGestures(Base):
    __tablename__ = table_name
    __table_args__ = {'schema': schema_name}  # Especifica el esquema aquí

    id = Column(Integer, primary_key=True)
    class_name = Column(String, nullable=False)

    # Definir 21 puntos (x, y) para landmarks de la mano
    for i in range(21):
        locals()[f'x_{i}'] = Column(Float, nullable=False)
        locals()[f'y_{i}'] = Column(Float, nullable=False)

# inspector gather information about the database
inspector = inspect(engine)
table_exists = inspector.has_table(table_name, schema=schema_name)

# check if the table exists
if table_exists:
    logger.info(f"La tabla '{table_name}' ya existe en el esquema '{schema_name}' en la BD.")
else:
    logger.info(f"La tabla '{table_name}' no existe en el esquema '{schema_name}'. Creando la tabla...")
    # Crear la tabla en la BD
    Base.metadata.create_all(engine, checkfirst=True)
    logger.info(f"Tabla '{table_name}' creada en el esquema '{schema_name}' en la BD.")

# Crear una sesion
Session = sessionmaker(bind=engine)
session = Session()