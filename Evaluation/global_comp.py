import pandas as pd
import os
import matplotlib.pyplot as plt

folders = {
    "Fixed": "Comparativa_Registros_Fixed",
    "Random": "Comparativa_Registros_Random"
}
csv_file = "reporte_completo.csv"
csv_global = "metricas_globales.csv"

def resumen_metricas(path, path_global):
    df = pd.read_csv(path)
    df_global = pd.read_csv(path_global)
    resumen = []
    for epoca in sorted(df['Épocas'].unique()):
        sub = df[df['Épocas'] == epoca]
        dsc_mean = sub['DSC'].mean()
        dsc_std = sub['DSC'].std()
        mae_img = sub['MAE_Imagen'].iloc[0]
        mae_seg = sub['MAE_Seg'].iloc[0]
        mag_def = sub['Magnitud_Deform'].iloc[0]
        # Buscar el DSC global para la época
        dsc_global = df_global[df_global['Épocas'] == epoca]['DSC_Global'].values
        dsc_global = dsc_global[0] if len(dsc_global) > 0 else None
        resumen.append({
            "Épocas": epoca,
            "DSC_mean_labels": dsc_mean,
            "DSC_std": dsc_std,
            "DSC_global": dsc_global,
            "MAE_Imagen": mae_img,
            "MAE_Seg": mae_seg,
            "Magnitud_Deform": mag_def
        })
    return pd.DataFrame(resumen)

resumenes = {}
for mode, folder in folders.items():
    path = os.path.join(folder, csv_file)
    path_global = os.path.join(folder, csv_global)
    resumen = resumen_metricas(path, path_global)
    resumenes[mode] = resumen
    print(f"\n=== {mode} ===")
    print(resumen.round(4))
    resumen.to_csv(f"resumen_metricas_{mode}.csv", index=False)

# Plot
plt.figure(figsize=(10, 6))
for mode, resumen in resumenes.items():
    plt.plot(resumen['Épocas'], resumen['DSC_mean_labels'], marker='o', label=f'{mode} - DSC mean labels')
    plt.plot(resumen['Épocas'], resumen['DSC_global'], marker='s', linestyle='--', label=f'{mode} - DSC global')

plt.xlabel('Épocas')
plt.ylabel('DSC')
plt.title('Evolución del DSC por época')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()