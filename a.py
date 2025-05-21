import os
from PIL import Image
import matplotlib.pyplot as plt

def plot_cams_grid(models, datasets, base_dir='output', cams_folder='cams', figsize_per_image=(4,4), save_to=None):
    n_rows = len(datasets)
    n_cols = len(models)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * figsize_per_image[0],
                                      n_rows * 2.8),
                             squeeze=False,
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    # Percorre cada célula da grade
    for i, (ds_name, class_name, img_name) in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i, j]
            img_path = os.path.join(base_dir, model, ds_name, cams_folder, img_name)
            if os.path.isfile(img_path):
                img = Image.open(img_path)
                ax.imshow(img)
            else:
                # Se não achar a imagem, deixa em branco
                ax.text(0.5, 0.5, 'N/A',
                        ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')

            # Título da coluna (somente na primeira linha)
            # if i == 0:
            #     ax.set_title(model, fontsize=14, pad=10)

        # Rótulo da linha (à esquerda da primeira coluna)
        # combinando dataset e classe
        # axes[i, 0].set_ylabel(f"{ds_name}\n({class_name})",
        #                       rotation=0, labelpad=60,
        #                       va='center', fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(
        left=0.25,    # já para o labelpad do ylabel
        wspace=0.05,  # gap horizontal pequeno
        hspace=0.05   # gap vertical pequeno
    )

    if save_to:
        plt.savefig(save_to, dpi=300, bbox_inches='tight')
        print(f"Figura salva em: {save_to}")
    else:
        plt.show()


if __name__ == "__main__":
    # Exemplo de uso
    models = [
        "COATNETB0",
        "COATNET_ABN_CF_GAP",
        # adicione quantos modelos quiser...
    ]

    datasets = [
        # (nome_da_pasta_do_dataset, classe, nome_do_arquivo_na_pasta cams)
        ("CR_2_CV",    "Benigno", "i1_1_11.png.png"),
        ("LA_4_CV",      "LA",      "i1_3_62.png.png"),
        ("LG_2_CV",      "LG",      "i1_0_26.png.png"),
        ("NHL_3_CV",       "CLL",     "i1_1_16.png.png"),
        ("UCSB_2_CV",  "Benigno", "i1_0_5.png.png"),
    ]

    # Se quiser salvar a figura em disco, passe save_to="grid_cam.png"
    plot_cams_grid(models, datasets, base_dir="output", save_to='a.png')
