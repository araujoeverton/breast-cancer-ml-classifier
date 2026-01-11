import os
import zipfile
import logging
import sys
from tqdm import tqdm

# --- 1. Configuração do Logger ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)


def extract_dataset(zip_path: str, extract_to: str = "data"):
    """
    Extrai um arquivo ZIP para o diretório de destino.
    """
    if not os.path.exists(zip_path):
        logger.error(f"Arquivo não encontrado: {zip_path}")
        return

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        logger.info(f"Diretório criado: {extract_to}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            logger.info(f"Extraindo {len(file_list)} arquivos de {os.path.basename(zip_path)}...")

            for file in tqdm(file_list, desc="Extraindo", unit="files"):
                zip_ref.extract(member=file, path=extract_to)

        logger.info(f"Sucesso! Arquivos extraídos em: {os.path.abspath(extract_to)}")

    except zipfile.BadZipFile:
        logger.error("Erro: O arquivo está corrompido ou não é um ZIP válido.")
    except Exception as e:
        logger.error(f"Erro inesperado na extração: {e}")


def list_directory_structure(start_path: str):
    """
    Lista a estrutura de pastas.
    """
    if not os.path.exists(start_path):
        logger.warning(f"Caminho não existe: {start_path}")
        return

    print(f"\nEstrutura de: {start_path}")
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files[:5]:
            print(f'{subindent}{f}')
        if len(files) > 5:
            print(f'{subindent}... ({len(files) - 5} outros arquivos)')


def download_from_kaggle(dataset_slug: str, output_path: str = "data"):
    """
    Baixa dataset usando API do Kaggle (Importação Tardia).
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logger.info(f"Conectando ao Kaggle para baixar {dataset_slug}...")

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()  # Aqui ele vai ler o os.environ que você definiu no notebook
    except (ImportError, OSError) as e:
        logger.error(f"Erro ao conectar com o Kaggle. Verifique se 'kaggle.json' está configurado.")
        logger.error(f"Detalhe do erro: {e}")
        return None

    try:
        api.dataset_download_files(dataset_slug, path=output_path, unzip=False, quiet=False)

        zip_name = dataset_slug.split('/')[-1] + ".zip"
        full_path = os.path.join(output_path, zip_name)

        logger.info(f"Download concluído: {full_path}")
        return full_path

    except Exception as e:
        logger.error(f"Erro durante download: {e}")
        return None

def download_and_extract(dataset_slug: str, data_dir: str = "data"):
    """
    Orquestra Download -> Extração.
    """
    zip_name = dataset_slug.split('/')[-1] + ".zip"
    zip_path = os.path.join(data_dir, zip_name)
    extract_path = os.path.join(data_dir, dataset_slug.split('/')[-1])

    if not os.path.exists(zip_path):
        logger.info(f"Arquivo {zip_name} não encontrado. Iniciando download...")
        zip_path_result = download_from_kaggle(dataset_slug, data_dir)
        if not zip_path_result:
            return None  # Para se o download falhar
    else:
        logger.info(f"Arquivo ZIP já existe: {zip_path}")

    if not os.path.exists(extract_path):
        logger.info(f"Iniciando extração para: {extract_path}")
        extract_dataset(zip_path, extract_path)
    else:
        logger.info(f"Dados já extraídos em: {extract_path}")

    return extract_path