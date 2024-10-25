import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# Função para ler metadados de um arquivo JSON
def read_metadata(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return None

# Função para calcular o tamanho dos arquivos em bytes
def get_file_size(file_path):
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0

# Função para atualizar as estatísticas do dataset com os valores de um metadado
def update_statistics(metadata, dir_path, stats):
    # Atualizar contagem para cada tonalidade
    key = metadata.get("key", "Unknown")
    stats["keys"][key] += 1

    # Atualizar contagem para cada andamento
    tempo = metadata.get("tempo", "Unknown")
    stats["tempos"][tempo] += 1

    # Atualizar contagem para cada assinatura de tempo
    time_signature = metadata.get("time_signature", "Unknown")
    stats["time_signatures"][time_signature] += 1

    # Atualizar a distribuição dos tipos de seções (ex. intro, verse, chorus)
    arrangement = metadata.get("arrangement", [])
    for section in arrangement:
        stats["section_distribution"][section] += 1

    # Armazenar a estrutura completa de seções como uma string
    section_tuple = tuple(arrangement)
    section_string = " -> ".join(arrangement)  # Converte a estrutura de tupla para uma string
    stats["section_structures"][section_string] += 1

    # Atualizar contagem de transições
    stats["total_transitions"] += len(metadata.get("transitions", []))

    # Atualizar número médio de compassos para cada seção
    measures = metadata.get("measures", {})
    for section, length in measures.items():
        stats["section_measures"][section].append(length)

    # Atualizar as informações de soundfonts utilizadas
    soundfonts = metadata.get("soundfonts", {})
    for part, soundfont in soundfonts.items():
        stats["soundfonts"][soundfont] += 1

    # Atualizar informações sobre pedalboards (efeitos e instrumentos)
    pedalboards = metadata.get("pedalboards", {})
    for instrument, effects in pedalboards.items():
        for effect in effects:
            effect_name = effect.get("name", "Unknown")
            stats["effects"][effect_name] += 1

    # Calcular a pontuação de musicalidade média
    musicality_score = metadata.get("musicality_score", 0.0)
    stats["musicality_scores"].append(musicality_score)

    # Calcular tamanho dos arquivos gerados na instância
    instance_files = os.listdir(dir_path)
    total_size = 0
    extension_sizes = defaultdict(int)

    for file_name in instance_files:
        file_path = os.path.join(dir_path, file_name)
        file_size = get_file_size(file_path)
        total_size += file_size

        # Separar por extensão
        _, extension = os.path.splitext(file_name)
        extension_sizes[extension] += file_size

    # Atualizar as estatísticas de tamanho
    stats["total_size"] += total_size
    stats["file_size_distribution"][dir_path] = total_size
    stats["total_instances"] += 1  # Incrementa a quantidade total de instâncias (músicas)

    # Atualizar tamanho médio por extensão
    for ext, size in extension_sizes.items():
        stats["average_size_per_extension"][ext].append(size)

# Função para calcular estatísticas finais e salvar em um arquivo JSON
def calculate_final_statistics(stats):
    # Calcular proporção de cada tonalidade
    total_keys = sum(stats["keys"].values())
    stats["keys_percentage"] = {k: (v / total_keys) * 100 for k, v in stats["keys"].items()}

    # Calcular proporção de cada andamento
    total_tempos = sum(stats["tempos"].values())
    stats["tempos_percentage"] = {k: (v / total_tempos) * 100 for k, v in stats["tempos"].items()}

    # Calcular proporção de cada assinatura de tempo
    total_signatures = sum(stats["time_signatures"].values())
    stats["time_signatures_percentage"] = {k: (v / total_signatures) * 100 for k, v in stats["time_signatures"].items()}

    # Calcular proporção de seções
    total_sections = sum(stats["section_distribution"].values())
    stats["section_distribution_percentage"] = {k: (v / total_sections) * 100 for k, v in stats["section_distribution"].items()}

    # Calcular número médio de compassos por seção
    stats["average_measures_per_section"] = {section: (sum(lengths) / len(lengths)) if lengths else 0
                                             for section, lengths in stats["section_measures"].items()}

    # Calcular média de pontuação de musicalidade
    if stats["musicality_scores"]:
        stats["average_musicality_score"] = sum(stats["musicality_scores"]) / len(stats["musicality_scores"])
    else:
        stats["average_musicality_score"] = 0.0

    # Calcular tamanho médio por extensão
    stats["average_size_per_extension_final"] = {ext: (sum(sizes) / len(sizes)) if sizes else 0
                                                 for ext, sizes in stats["average_size_per_extension"].items()}

    # Calcular a estrutura de seção mais comum
    if stats["section_structures"]:
        most_common_structure = max(stats["section_structures"], key=stats["section_structures"].get)
        stats["most_common_structure"] = most_common_structure
        stats["most_common_structure_count"] = stats["section_structures"][most_common_structure]

    # Calcular média do tamanho das instâncias
    if stats["total_instances"] > 0:
        stats["average_instance_size"] = stats["total_size"] / stats["total_instances"]
    else:
        stats["average_instance_size"] = 0.0

    # Salvar estatísticas em um arquivo JSON
    with open('dataset_statistics.json', 'w') as f:
        json.dump(stats, f, indent=4)

# Função principal para varrer diretórios e calcular estatísticas
def main():
    # Diretório raiz onde os subdiretórios de instâncias estão localizados
    root_dir = './datamount'

    # Estrutura de dicionário para armazenar estatísticas
    stats = {
        "keys": defaultdict(int),
        "tempos": defaultdict(int),
        "time_signatures": defaultdict(int),
        "section_distribution": defaultdict(int),
        "section_structures": defaultdict(int),
        "total_transitions": 0,
        "section_measures": defaultdict(list),
        "soundfonts": defaultdict(int),
        "effects": defaultdict(int),
        "musicality_scores": [],
        "total_size": 0,
        "file_size_distribution": defaultdict(int),
        "average_size_per_extension": defaultdict(list),
        "total_instances": 0,  # Adicionado para contar a quantidade de instâncias
        "average_musicality_score": 0.0  # Média de musicality score adicionada como campo separado
    }

    # Varre todos os subdiretórios na raiz
    for dir_name in tqdm(os.listdir(root_dir), desc="Processando diretórios"):
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):
            # Caminho do arquivo JSON de metadados dentro do diretório da instância, usando o nome do diretório
            metadata_path = os.path.join(dir_path, f'{dir_name}.json')
            metadata = read_metadata(metadata_path)
            if metadata:
                update_statistics(metadata, dir_path, stats)

    # Calcular estatísticas finais e salvar no arquivo de saída
    calculate_final_statistics(stats)
    print("Estatísticas do dataset salvas em 'dataset_statistics.json'.")

# Executar script
if __name__ == "__main__":
    main()
