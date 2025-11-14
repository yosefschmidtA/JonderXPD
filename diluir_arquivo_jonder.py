import pandas as pd
import numpy as np
import warnings

# Suprimir avisos
warnings.filterwarnings('ignore')

# Caminho do arquivo CSV
csv_file_path = "csv_dados_XPD.csv"

print(f"Iniciando o processamento do arquivo: {csv_file_path}")
print("Lógica CORRIGIDA (header=0, skiprows=[1]).")
print("Theta = Coluna 1 (0-52), Phi = Cabeçalho (76-128)")

try:
    # 1. Carregar o CSV
    df_new = pd.read_csv(
        csv_file_path,
        delimiter=';',
        decimal=',',
        encoding='latin-1',
        header=0,
        skiprows=[1]  # Pula a segunda linha inútil
    )

    print("Arquivo CSV carregado.")

    # 2. Renomear a primeira coluna para 'Theta' (CORRIGIDO)
    df_new.rename(columns={df_new.columns[0]: 'Theta'}, inplace=True)

    # Converter Theta para numérico
    df_new['Theta'] = pd.to_numeric(df_new['Theta'])

    # 3. "Derretimento" Manual
    data_to_melt = []

    current_phi = np.nan  # (CORRIGIDO)
    col_index = 0

    # Iterar por todas as colunas, exceto a primeira ('Theta')
    for col_name in df_new.columns[1:]:

        # Tenta converter o nome da coluna para um número.
        # Se funcionar, é um novo Phi (e o Componente 1).
        try:
            current_phi = float(col_name)  # (CORRIGIDO)
            col_index = 1
            comp_name = "C1"

        except ValueError:
            if str(col_name).startswith('Unnamed:'):
                col_index += 1
                comp_name = f"C{col_index}"
            else:
                continue

        # Pular colunas de espaçamento
        if df_new[col_name].isnull().all():
            col_index = 0
            continue

        # Criar um dataframe temporário
        # 'Theta' já é a coluna correta
        temp_df = df_new[['Theta', col_name]].copy()

        temp_df['Phi'] = current_phi  # (CORRIGIDO)
        temp_df['Component'] = comp_name

        temp_df.rename(columns={col_name: 'Intensity'}, inplace=True)
        data_to_melt.append(temp_df)

    # 4. Concatenar
    df_long = pd.concat(data_to_melt, ignore_index=True)

    # 5. Limpar NAs finais e reordenar
    df_long = df_long.dropna(subset=['Intensity'])
    df_long['Intensity'] = pd.to_numeric(df_long['Intensity'])

    # A estrutura final (antes de salvar) está correta
    df_final = df_long[['Theta', 'Phi', 'Component', 'Intensity']]

    print("\nDados transformados para o formato 'longo' (Theta, Phi, Component, Intensity).")

    # 6. Salvar os 4 arquivos de componente
    components = df_final['Component'].unique()
    print(f"Componentes encontrados (assumidos): {', '.join(components)}")

    output_files = []

    for comp in components:
        df_comp = df_final[df_final['Component'] == comp]

        # Selecionar e renomear as colunas
        # Agora 'Theta' -> 'theta' e 'Phi' -> 'phi' está correto
        df_to_save = df_comp[['Theta', 'Phi', 'Intensity']]
        df_to_save = df_to_save.rename(columns={
            'Theta': 'theta',
            'Phi': 'phi',
            'Intensity': 'intensity'
        })

        output_file_name = f"../saidatpintensity_{comp}.txt"

        # Salvar o arquivo
        df_to_save.to_csv(
            output_file_name,
            sep=' ',
            index=False,
            float_format='%.1f'
        )
        output_files.append(output_file_name)

    print("\nProcessamento concluído com sucesso!")
    print("Os seguintes arquivos (CORRIGIDOS) foram gerados:")
    for f in output_files:
        print(f)

except Exception as e:
    print(f"\nOcorreu um erro durante o processamento:")
    print(e)
    import traceback

    traceback.print_exc()